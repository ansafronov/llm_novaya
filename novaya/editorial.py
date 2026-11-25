"""Prepare paired editorial examples via a local llama.cpp server."""
import argparse
import json
import re
import time
from functools import partial
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .data import evenly, file_sha256, normalize, parse_article, read_rows, write_json, write_rows


CLEAN = """Верни JSON {"remove":[{"id":1,"reason":"caption"}]}.
Помечай только целые служебные абзацы: caption — подпись к фото,
media — указатель на медиа, credit — отдельное авторство или фотокредит,
related — ссылки на другие материалы. Сохраняй содержательный текст,
подзаголовки и авторство цитат. При сомнении оставь абзац."""
FACTS = """Верни JSON {"blocks":[{"ids":[1,2],"anchors":["1.1","1.2","2.1"]}]}.
Выбери до четырёх непересекающихся самостоятельных фрагментов из 1–3
соседних абзацев, 40–160 слов каждый. anchors — от 3 до 8 разных ID
предложений, которые вместе покрывают существенные утверждения фрагмента.
Если предложений меньше трёх, укажи все.
Возвращай только ID, без цитат. Пропускай фрагменты с неясным контекстом,
сатирой, подписями к изображениям или анонсами других материалов."""
REWRITE = """Верни JSON {"texts":["..."]}. Перепиши каждое утверждение
отдельно простым русским языком в том же порядке. Используй исходное
предложение и соседние предложения, чтобы раскрыть местоимения и сохранить
атрибуцию, степень уверенности и стадию события. Если источник говорит об
оценке, плане или попытке, прямо обозначь это в результате. Сохрани имена,
числа, даты и условия. Поле context помогает понять, кто или что
подразумевается, и отличить авторскую оценку, интервью или сатиру.
Не переноси оттуда новые события, оценки или причинно-следственные связи.
Если в архивном тексте есть «сейчас», «недавно» или срок без начальной
даты, укажи дату публикации как точку отсчёта. Не выдавай эту дату за дату события.
При каждом пересказе проверь, кто именно действовал или говорил, что
уже произошло, что только возможно, при каком условии и в каком объёме.
Например, «по оценке инженеров, большинство пилотных устройств сможет
работать не менее суток после доработки» можно пересказать так:
«Инженеры считают, что после доработки большая часть пилотных устройств
сможет работать как минимум сутки». Здесь пока нет готового результата."""
LEAD = ("Напиши начало заметки для «Новой газеты» по этим фактам. "
        "Лид должен состоять из 2–3 предложений. "
        "Не добавляй новых утверждений.")
CHECK_CARD = """Проверь каждый пересказ отдельно. Сначала найди конкретные
расхождения между пересказом, цитатой и исходным предложением. Сравни
субъекта, действие, числа, автора мнения, степень уверенности, условия
и стадию события. Затем перечисли существенные утверждения исходного
фрагмента, которые отсутствуют в карточке.
Считай ошибкой изменение смысла. Допускай синонимы, другую структуру
предложения и опущенные выразительные детали, если смысл сохранён.
Если сама карточка выдаёт сатиру за факт, укажи это как ошибку.
Проверь, что читатель одной карточки понимает, о каком событии или
участнике идёт речь. Поле context разрешает уточнить ссылки вроде
«эта жалоба» или «он». Новые утверждения из context добавлять нельзя.
Для относительных сроков проверь, что точка отсчёта понятна.
Верни JSON {"errors":["описание расхождения"],"missing":["пропущенный факт"]}.
Опиши каждый пункт одной короткой фразой. Пустые списки означают,
что смысловых расхождений и существенных пропусков нет."""
CHECK_LEAD = """Сверь лид с карточкой и исходными предложениями. Верни JSON
{"ok":true}, если каждое утверждение лида следует и из карточки, и из
источника. Проверь, кто совершил действие или высказал оценку, сохранились
ли оговорки и стадия события. Лид может опустить второстепенные факты.
Если хотя бы одно утверждение расходится с источником или выходит за
пределы карточки, верни {"ok":false}. Поле context используй только для
уточнения, к кому или чему отсылает исходный фрагмент."""
FRAGMENT = ("Напиши короткий фрагмент заметки для «Новой газеты» по этим фактам. "
            "Не добавляй новых утверждений.")


def sentences(text):
    text = text.strip()
    abbreviation_spaces = {
        match.start() + offset
        for pattern in (r'\bи\.\s*о\.\s+(?=[а-яё])',
                        r'\d\s+тыс\.\s+(?=[а-яё])',
                        r'\b(?:[А-ЯЁ]\.\s*){1,2}(?=[А-ЯЁ][а-яё]+)')
        for match in re.finditer(pattern, text)
        for offset, char in enumerate(match.group())
        if char.isspace() and offset and match.group()[offset - 1] == '.'
    }
    chunks, start = [], 0
    for boundary in re.finditer(r'(?<=[.!?…])\s+', text):
        if boundary.start() not in abbreviation_spaces:
            chunks.append(text[start:boundary.start()])
            start = boundary.end()
    return chunks + [text[start:]]


def request(port, instruction, payload, validate, *, thinking=False):
    body = dict(messages=[dict(role='system', content=instruction),
                          dict(role='user', content=json.dumps(payload, ensure_ascii=False))],
                temperature=0, max_tokens=2048 if thinking else 1000, seed=42,
                chat_template_kwargs={'enable_thinking': thinking},
                response_format={'type': 'json_object'})
    for attempt in range(2):
        try:
            with urlopen(Request(f'http://127.0.0.1:{port}/v1/chat/completions',
                                 data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'}), timeout=180) as response:
                result = json.load(response)['choices'][0]['message']['content']
            return validate(json.loads(result))
        except HTTPError as error:
            if error.code not in (400, 413, 422):
                raise
            if attempt:
                raise ValueError(f'request_rejected_http_{error.code}') from error
        except (ValueError, KeyError, TypeError, IndexError) as error:
            if attempt:
                raise ValueError(f'invalid_model_response: {error}') from error
            body['messages'].append(dict(role='user', content=(
                f'Предыдущий ответ отклонён: {error}. Исправь и верни только JSON.')))


def chunks(paragraphs):
    current = {}
    for index, paragraph in paragraphs.items():
        if current and (len(current) >= 8 or sum(map(len, current.values())) + len(paragraph) > 3200):
            yield current
            current = {}
        if len(paragraph) <= 3200:
            current[index] = paragraph
    if current:
        yield current


def safe_removal(paragraph, reason):
    line = paragraph.strip().strip('*_ ').strip()
    if '\n' in line or len(line.split()) > 20:
        return False
    if reason == 'caption' and line.casefold() == 'фото':
        return True
    if reason == 'caption' and paragraph.strip().startswith('*') and re.search(
        r'(?:\.\s*Фото из соцсетей|/\s*Фотохроника ТАСС)\.?$', line, re.I
    ):
        return True
    if reason == 'credit' and re.fullmatch(
        r'[А-ЯЁ][а-яёА-ЯЁ -]+\s+—\s*\**\s*специально для [«„]Новой[»“]\.?',
        line, re.I
    ):
        return True
    if reason == 'related' and re.fullmatch(
        r'link:\*{0,2}\s*\[[^\]]+\]\(\d{4}/\d{2}/\d{2}/[^()]+\.md\)',
        line, re.I
    ):
        return True
    markers = {
        'caption': r'(?:фото|иллюстрация|рисунок|на фото)\s*[:：]',
        'credit': r'(?:автор|фотограф|фото|фотокредит)\s*[:：]',
        'media': r'(?:видео|галерея|аудио)\s*[:：]',
        'related': r'читайте также\s*[:：]',
    }
    return bool(re.match(markers[reason], line, re.I))


def standalone_media_credit(paragraph):
    paragraph = paragraph.strip()
    if re.fullmatch(
        r'\*[^*\n]{1,180}\.\s*(?:Кадр из видео Goldman Group|Фото ЗС Красноярского края)\.?\*',
        paragraph,
    ):
        return True
    return re.fullmatch(
        r'\*(?:[А-ЯЁ][а-яё-]+(?: [А-ЯЁ][а-яё-]+){1,2}\.\s*'
        r'(?i:Фото:\s*пресс-служба суда)|(?i:Скрин:\s*YouTube))\.?\*',
        paragraph,
    ) is not None


def verdict(result):
    if type(result['ok']) is not bool:
        raise ValueError('expected boolean verdict')
    return result['ok']


def fallback_blocks(chunk, indexed):
    ids = list(chunk)
    next_start = 0
    for start in range(len(ids)):
        if start < next_start:
            continue
        for end in range(start + 1, min(start + 3, len(ids)) + 1):
            block_ids = ids[start:end]
            if block_ids[-1] - block_ids[0] != len(block_ids) - 1:
                break
            size = sum(len(chunk[index].split()) for index in block_ids)
            anchors = [anchor for index in block_ids for anchor in indexed[index]]
            if size > 160 or len(anchors) > 8:
                break
            if size >= 40:
                yield dict(ids=block_ids, anchors=anchors)
                next_start = end
                break


def card_verdict(result):
    for field in ('errors', 'missing'):
        if not isinstance(result[field], list) or any(
            not isinstance(item, str) or not item.strip() for item in result[field]
        ):
            raise ValueError('expected lists of concrete discrepancies')
    return result


def prepare(row, archive, port, issues=None, *, thinking=False, fixed_windows=False):
    ask = partial(request, thinking=True) if thinking else request
    parsed = parse_article((archive / row['source']).read_text(encoding='utf-8'), row['source'])
    paragraphs = dict(enumerate(re.split(r'\n\s*\n', parsed['text']), 1))
    if issues is None:
        issues = []
    issues.extend(dict(stage='chunk', ids=[index], reason='paragraph_too_long')
                  for index, paragraph in paragraphs.items() if len(paragraph) > 3200)
    removed = [
        dict(id=index, reason='caption', text=paragraph)
        for index, paragraph in paragraphs.items()
        if standalone_media_credit(paragraph)
    ]
    for item in removed:
        del paragraphs[item['id']]
    for chunk in chunks(paragraphs.copy()):
        def validate_clean(result):
            items = result['remove']
            ids = [item['id'] for item in items]
            if len(set(ids)) != len(ids) or any(
                type(item['id']) is not int or item['id'] not in chunk
                or item['reason'] not in ('caption', 'credit', 'media', 'related') for item in items
            ):
                raise ValueError('invalid removal')
            return items

        try:
            removals = ask(port, CLEAN, chunk, validate_clean)
        except ValueError as error:
            issues.append(dict(stage='clean', ids=list(chunk), reason=str(error)))
            continue
        for item in removals:
            paragraph = paragraphs[item['id']].lstrip()
            if not safe_removal(paragraph, item['reason']):
                issues.append(dict(stage='clean', ids=[item['id']], reason='protected_paragraph',
                                   text=paragraph, proposed_reason=item['reason']))
                continue
            removed.append({**item, 'text': paragraphs.pop(item['id'])})

    pairs = []
    for chunk in chunks(paragraphs):
        indexed = {index: {f'{index}.{offset}': sentence for offset, sentence
                           in enumerate(sentences(paragraph), 1)} for index, paragraph in chunk.items()}

        def validate_blocks(result):
            used = set()
            blocks = result['blocks']
            if not isinstance(blocks, list) or len(blocks) > 4:
                raise ValueError('expected at most four blocks')
            accepted, rejected = [], []
            order = list(chunk)
            for block in blocks:
                ids = block.get('ids') if isinstance(block, dict) else None
                if (not isinstance(ids, list) or not 1 <= len(ids) <= 3
                    or any(type(index) is not int or index not in chunk for index in ids)
                    or ids != order[order.index(ids[0]):][:len(ids)] or used.intersection(ids)):
                    rejected.append(dict(ids=list(chunk), reason='invalid block'))
                    continue
                source = {key: value for index in ids for key, value in indexed[index].items()}
                anchors = block.get('anchors')
                if not isinstance(anchors, list) or not min(3, len(source)) <= len(anchors) <= 8 or any(
                    not isinstance(anchor, str) or anchor not in source for anchor in anchors
                ) or len(set(anchors)) != len(anchors):
                    rejected.append(dict(ids=ids, reason='invalid sentence anchors'))
                    continue
                used.update(ids)
                accepted.append(block)
            if rejected and not accepted:
                raise ValueError('; '.join(item['reason'] for item in rejected))
            return accepted, rejected

        blocks, rejected = fallback_blocks(chunk, indexed), []
        if not fixed_windows:
            try:
                blocks, rejected = ask(port, FACTS, indexed, validate_blocks)
            except ValueError as error:
                issues.append(dict(stage='facts', ids=list(chunk), reason=str(error)))
        issues.extend(dict(stage='facts', **item) for item in rejected)
        for block in blocks:
            target = '\n\n'.join(chunk[index] for index in block['ids'])
            if not 40 <= len(target.split()) <= 160:
                issues.append(dict(stage='fragment', ids=block['ids'], reason='invalid_length'))
                continue
            evidence = {key: value for index in block['ids'] for key, value in indexed[index].items()}
            context = dict(title=parsed.get('title', ''), genre=parsed.get('genre', ''),
                           publication_date=parsed.get('date', ''),
                           preceding=paragraphs.get(block['ids'][0] - 1, '')[-600:],
                           following=paragraphs.get(block['ids'][-1] + 1, '')[:600])

            claims = [dict(quote=evidence[anchor], sources=[anchor]) for anchor in block['anchors']]

            def validate_rewrites(result):
                texts = result['texts']
                if not isinstance(texts, list) or len(texts) != len(claims) or any(
                    not isinstance(item, str) or not item.strip() for item in texts
                ):
                    raise ValueError('expected one rewrite per claim')
                texts = [item.strip() for item in texts]
                if copied(texts, target) or long_copied_claim(texts, target):
                    raise ValueError('copied_input: reformulate the copied passage in simpler words')
                return texts

            try:
                facts = ask(port, REWRITE, dict(evidence=evidence, context=context, claims=claims),
                                validate_rewrites)
            except ValueError as error:
                issues.append(dict(stage='rewrite', ids=block['ids'], reason=str(error)))
                continue
            grounded = [dict(fact, text=rewritten) for fact, rewritten in zip(claims, facts)]
            card = '\n'.join('- ' + fact for fact in facts)
            try:
                checked = ask(port, CHECK_CARD, dict(evidence=evidence, context=context, facts=grounded),
                                  card_verdict)
            except ValueError as error:
                issues.append(dict(stage='card_check', ids=block['ids'], reason=str(error)))
                continue
            if checked['errors'] or checked['missing']:
                issues.append(dict(stage='card_check', ids=block['ids'],
                                   reason='unsupported_or_incomplete_card', **checked))
                continue

            def validate_lead(result):
                if not isinstance(result['lead'], str):
                    raise ValueError('expected lead text')
                answer = result['lead'].strip()
                if not 2 <= len([part for part in sentences(answer) if part]) <= 3:
                    raise ValueError('expected a lead of two or three sentences')
                if copied(facts, answer):
                    raise ValueError('lead copied a fact verbatim')
                return answer

            try:
                lead = ask(port, LEAD + ' Верни только JSON {"lead":"..."}.', card, validate_lead)
                if not ask(port, CHECK_LEAD, dict(card=card, evidence=evidence, context=context, lead=lead),
                           verdict):
                    raise ValueError('unsupported_lead')
            except ValueError as error:
                issues.append(dict(stage='lead', ids=block['ids'], reason=str(error)))
                lead = None
            pairs.append(dict(article_id=row['id'], source=row['source'], ids=block['ids'],
                              facts=grounded, evidence=evidence, context=context,
                              fragment=example(FRAGMENT, card, target),
                              lead=example(LEAD, card, lead) if lead else None))
            if len(pairs) == 4:
                return pairs, removed
    return pairs, removed


def copied(facts, target):
    original = {normalize(sentence).rstrip('.!?…') for sentence in sentences(target)}
    return any(original.intersection(normalize(sentence).rstrip('.!?…') for sentence in sentences(fact))
               for fact in facts)


def long_copied_claim(facts, target):
    size = 16
    words = re.findall(r'\w+', target.casefold())
    excerpts = {tuple(words[start:start + size]) for start in range(len(words) - size + 1)}
    for fact in facts:
        words = re.findall(r'\w+', fact.casefold())
        windows = [tuple(words[start:start + size]) for start in range(len(words) - size + 1)]
        if windows and sum(window in excerpts for window in windows) / len(windows) >= 0.75:
            return True
    return False


def example(instruction, card, text):
    return dict(messages=[dict(role='user', content=instruction + '\n\n' + card),
                          dict(role='assistant', content=text)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--labeler', default='Qwen/Qwen3-14B-GGUF Q4_K_M',
                        help='Model identifier for the local server')
    parser.add_argument('--split', choices=('train', 'validation'), default='train')
    parser.add_argument('--thinking', action='store_true', help='Enable the model reasoning template')
    parser.add_argument('--fixed-windows', action='store_true',
                        help='Select source paragraphs in code before paraphrasing')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--articles', type=int, default=100)
    selection.add_argument('--article-id', action='append', default=[],
                           help='Run only the specified article IDs, in the given order')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.articles < 1:
        parser.error('--articles must be positive')
    split_path = Path('data/prepared') / (args.split + '.jsonl')
    rows = read_rows(split_path)
    if args.article_id:
        by_id = {row['id']: row for row in rows}
        if len(set(args.article_id)) != len(args.article_id):
            parser.error('Duplicate article ID')
        missing = set(args.article_id) - by_id.keys()
        if missing:
            parser.error('Article IDs outside the selected split: ' + ', '.join(sorted(missing)))
        selected = [by_id[article_id] for article_id in args.article_id]
    else:
        selected = evenly(rows, args.articles) if args.articles > 1 else rows[:1]
    args.output.mkdir(parents=True)
    settings = dict(split=args.split, split_sha256=file_sha256(split_path), labeler=args.labeler,
                    thinking=args.thinking,
                    selection='consecutive_paragraphs' if args.fixed_windows
                    else 'model_with_consecutive_paragraphs_fallback',
                    selected_article_ids=[row['id'] for row in selected],
                    prompts=dict(clean=CLEAN, facts=FACTS, lead=LEAD, fragment=FRAGMENT,
                                 rewrite=REWRITE, check_card=CHECK_CARD, check_lead=CHECK_LEAD))
    write_json(args.output / 'settings.json', settings)
    pairs, audit = [], []
    with (args.output / 'audit.jsonl').open('w') as log, (args.output / 'pairs.jsonl').open('w') as pair_log:
        for row in selected:
            start = time.monotonic()
            issues = []
            try:
                examples, removed = prepare(row, Path('data/archive'), args.port, issues,
                                            thinking=args.thinking, fixed_windows=args.fixed_windows)
                pairs.extend(examples)
                for pair in examples:
                    print(json.dumps(pair, ensure_ascii=False), file=pair_log)
                pair_log.flush()
                result = dict(pairs=len(examples), removed=removed)
            except ValueError as error:
                result = dict(pairs=0, error=str(error))
            record = dict(article_id=row['id'], source=row['source'],
                          seconds=time.monotonic() - start, issues=issues, **result)
            audit.append(record)
            print(json.dumps(record, ensure_ascii=False), file=log, flush=True)
    for variant in ('fragment', 'lead'):
        write_rows(args.output / (variant + '.jsonl'),
                   ({'article_id': pair['article_id'], **pair[variant]} for pair in pairs if pair[variant]))
    grouped = {}
    for pair in pairs:
        grouped.setdefault(pair['article_id'], []).append(pair)
    review = [group[index] for index in range(4) for group in grouped.values() if index < len(group)]
    write_rows(args.output / 'review.jsonl', review[:30])
    write_json(args.output / 'report.json', dict(**settings,
        articles=len(audit), fragments=len(pairs),
        leads=sum(pair['lead'] is not None for pair in pairs), seconds=sum(row['seconds'] for row in audit)))


if __name__ == '__main__':
    main()
