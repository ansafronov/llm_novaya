"""Build editorial training candidates from expressive archive prose and dry source facts."""
import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .data import file_sha256, normalize, parse_article, read_rows, write_json
from .editorial import example, request, standalone_media_credit

SELECT = """Ты отбираешь образцы языка «Новой газеты» для редакционного корпуса.
Найди один самостоятельный фрагмент из 1–3 соседних абзацев (40–160 слов).
Важны конкретная сцена или наблюдение, авторский голос, ритм, контраст, ирония,
характерная деталь или выразительная аргументация. Сухая заметка, список,
подпись, анонс и обрывок реплики не подходят. Не принимай необычную тему за
выразительный язык. Фрагмент должен быть понятен с минимальным контекстом.
Текст дан нумерованными абзацами. Выбери один из допустимых диапазонов:
первое и последнее числа — номера первого и последнего абзацев. Верни
{"range":[4,6]} либо {"range":null}, если подходящего отрывка нет."""

FACTS = """Подготовь нейтральное редакционное задание по выбранному фрагменту.
Перечисли сухими пунктами все сведения, без которых нельзя написать этот
фрагмент: кто, что сделал или сказал, при каких обстоятельствах, когда и где,
числа, порядок событий, конкретные детали сцены и существенные связи между ними.
Сохрани автора каждой оценки, версии, реплики или предположения. Если автор
сопоставляет две вещи или использует художественный образ, укажи простыми
словами, что с чем сопоставлено и что автор этим говорит. Не выдумывай объяснение.
Не оценивай стиль, не пиши лид и не повторяй выразительные обороты исходника.
Можно сохранять точные имена, названия, числа и обязательные формулировки цитат,
если важны сами слова говорящего. Соседний контекст только раскрывает имена,
отсылки и даты. Не вноси из него отдельные сведения, не подменяй им фрагмент.
Верни JSON {"facts":["один пункт","другой пункт"]}."""

TASK = 'Напиши фрагмент заметки для «Новой газеты» по этим фактам.'

LINK = re.compile(r'!?\[([^\]\n]*)\]\((?:[^()\n]|\([^()\n]*\))*\)')
HEADING = re.compile(r'^\s{0,3}#{1,6}\s|^\s{0,3}(?:=+|-{2,})\s*$')
SERVICE = re.compile(
    r'^\s*[*_> ]*(?:(?:фото|скрин(?:шот)?|кадр из|иллюстрация|читайте также|'
    r'смотрите также|подписывайтесь|слушайте|видео|автор)\s*[:—.]|'
    r'(?:18|16|12)\+(?:\s|$)|настоящий материал \(информация\))', re.I,
)
BARE_URL = re.compile(r'<?https?://[^\s<>]+>?')

def inline_prose(line):
    line = LINK.sub(lambda match: '' if match[0].startswith('!') else match[1], line)
    line = BARE_URL.sub(lambda match: re.search(r'[.,;:!?]+$', match[0])[0]
                      if re.search(r'[.,;:!?]+$', match[0]) else '', line)
    line = re.sub(r'^\s*(?:>\s*)+', '', line)
    for marker in ('**', '__', '*', '_'):
        delimiter = re.escape(marker[0])
        token = re.escape(marker)
        pattern = (rf'(?<!{delimiter}){token}(?!{delimiter}|\s)'
                   rf'(.+?)(?<!{delimiter})(?<=\S){token}(?!{delimiter})')
        line = re.sub(pattern, r'\1', line)
    return re.sub(r'[ \t]{2,}', ' ', line).strip()

def clean_paragraph(text, title):
    """Remove archive markup while keeping wording, punctuation and paragraph IDs."""
    if standalone_media_credit(text) or SERVICE.match(text):
        return ''
    lines = text.splitlines()
    cleaned = []
    for index, line in enumerate(lines):
        if HEADING.match(line) or SERVICE.match(line):
            continue
        if index + 1 < len(lines) and HEADING.fullmatch(lines[index + 1]):
            continue
        prose = inline_prose(line)
        if prose and normalize(prose) != normalize(title):
            cleaned.append(prose)
    return '\n'.join(cleaned)

def eligible_spans(paragraphs):
    choices = set()
    for first in paragraphs:
        count = 0
        for last in range(first, first + 3):
            if last not in paragraphs:
                break
            count += len(paragraphs[last].split())
            if count > 160:
                break
            if count >= 40:
                choices.add((first, last))
    return choices


def article_chunks(paragraphs, limit):
    """Keep paragraph boundaries and send each paragraph once."""
    chunk, size = {}, 0
    for index, text in paragraphs.items():
        if chunk and size + len(text) > limit:
            yield chunk
            chunk, size = {}, 0
        chunk[index] = text
        size += len(text)
    if chunk:
        yield chunk


def validate_facts(result):
    facts = result['facts']
    if not isinstance(facts, list) or not facts or any(
            not isinstance(fact, str) or not fact.strip() for fact in facts):
        raise ValueError('expected a nonempty list of fact strings')
    return [fact.strip() for fact in facts]

def process(row, port, chunk_chars=24000):
    source = row['source']
    article = parse_article((Path('data/archive') / source).read_text(), source)
    paragraphs = {i: text for i, raw in enumerate(re.split(r'\n\s*\n', article['text']), 1)
                  if (text := clean_paragraph(raw, article['title']))}
    record = dict(position=row['position'], article_id=row['article_id'], source=source,
                  title=article['title'], publication_date=article['date'], candidates=0,
                  selector_requests=0, selector_input_chars=0)

    metadata = dict(publication_date=article['date'], genre=article['genre'])
    selected = None
    errors = []
    start = time.monotonic()
    for chunk in article_chunks(paragraphs, chunk_chars):
        options = eligible_spans(chunk)
        record['candidates'] += len(options)
        if not options:
            continue

        def validate_selection(result):
            span = result['range']
            if span is None:
                return None
            if (not isinstance(span, list) or len(span) != 2
                    or any(type(index) is not int for index in span) or tuple(span) not in options):
                raise ValueError('choose one allowed range or null')
            return span

        payload = dict(**metadata, paragraphs=chunk, allowed_ranges=sorted(options))
        record['selector_requests'] += 1
        record['selector_input_chars'] += sum(map(len, chunk.values()))
        try:
            selected = request(port, SELECT, payload, validate_selection)
        except ValueError as error:
            errors.append(str(error))
        if selected is not None:
            break
    record['selection_seconds'] = round(time.monotonic() - start, 3)
    if selected is None:
        if errors:
            return dict(record, status='selection_error', errors=errors)
        return dict(record, status='skipped' if record['candidates'] else 'no_windows')
    first, last = selected
    ids = list(range(first, last + 1))
    target = '\n\n'.join(paragraphs[index] for index in ids)
    record.update(ids=ids, target=target)
    payload = dict(**metadata, fragment=target,
                   context_before=paragraphs.get(ids[0] - 1, '')[-800:],
                   context_after=paragraphs.get(ids[-1] + 1, '')[:800])
    try:
        facts = request(port, FACTS, payload, validate_facts)
    except ValueError as error:
        return dict(record, status='facts_error', error=str(error))
    return dict(record, status='candidate', facts=facts,
                **example(TASK, '\n'.join('- ' + fact for fact in facts), target))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--port', type=int, default=8083)
    parser.add_argument('--chunk-chars', type=int, default=24000)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if args.chunk_chars <= 0:
        parser.error('--chunk-chars must be positive')
    if args.workers <= 0:
        parser.error('--workers must be positive')
    rows = read_rows(args.sample)
    args.output.mkdir(parents=True, exist_ok=True)
    def timed_process(row):
        start = time.monotonic()
        record = process(row, args.port, args.chunk_chars)
        record['seconds'] = round(time.monotonic() - start, 3)
        return record

    with ((args.output / 'candidates.jsonl').open('x') as output,
          ThreadPoolExecutor(max_workers=args.workers) as executor):
        write_json(args.output / 'settings.json', dict(sample_sha256=file_sha256(args.sample),
                   model='Qwen3.8-27B UD-IQ4_XS', selector_prompt=SELECT, facts_prompt=FACTS,
                   chunk_chars=args.chunk_chars, workers=args.workers,
                   selection='nonoverlapping_paragraph_chunks'))
        for record in executor.map(timed_process, rows):
            print(json.dumps(record, ensure_ascii=False), file=output, flush=True)

if __name__ == '__main__':
    main()
