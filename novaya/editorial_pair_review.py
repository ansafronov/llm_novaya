"""Review author excerpts and synthetic leads against their shared facts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from .data import file_sha256, read_rows, write_json
from .editorial import request
from .editorial_style_review import PROMPT, validate

LEAD_PROMPT = """Проверь лид для «Новой газеты» по сухой карточке и исходному отрывку.
style — лид передаёт главное событие или тезис отрывка и читается как
самостоятельное начало заметки. Выразительность колонки не обязательна.
clean — нет заголовка, подписи, Markdown, URL и служебных вставок.
facts — каждое существенное утверждение лида опирается и на карточку, и на
отрывок. Сохранены атрибуция, условия и степень уверенности. Отсутствие
второстепенных деталей в лиде допустимо, дословное совпадение тоже.
Верни JSON {"style":true,"clean":true,"facts":true,"reason":"конкретное основание"}."""


def unique_rows(paths):
    records = {}
    for path in paths:
        for row in read_rows(path):
            article_id = row['article_id']
            if article_id in records:
                raise ValueError(f'Duplicate article {article_id} in {path}')
            records[article_id] = row
    return records


def review(port, prompt, payload):
    try:
        verdict = request(port, prompt, payload, validate)
    except ValueError as error:
        return dict(accepted=False, review_error=str(error))
    return dict(verdict, accepted=all(verdict[key] for key in ('style', 'clean', 'facts')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', type=Path, nargs='+', required=True)
    parser.add_argument('--leads', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8083)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()

    sources = unique_rows(args.candidates)
    candidates = {key: row for key, row in sources.items() if row['status'] == 'candidate'}
    leads = unique_rows([args.leads])
    if candidates.keys() != leads.keys():
        parser.error('Lead output must cover exactly the candidate articles, including failures')

    def check(article_id):
        row = candidates[article_id]
        lead = leads[article_id]
        if (lead['source'], lead['position']) != (row['source'], row['position']):
            raise ValueError(f'Lead provenance mismatch for {article_id}')
        payload = {key: row[key] for key in ('target', 'facts', 'publication_date')}
        fragment = review(args.port, PROMPT, payload)
        if lead['status'] == 'candidate':
            lead_review = review(args.port, LEAD_PROMPT, dict(payload, lead=lead['lead']))
        else:
            lead_review = dict(accepted=False, reason=lead['status'])
        return dict(article_id=article_id, position=row['position'], fragment=fragment,
                    lead=lead_review, accepted=fragment['accepted'] and lead_review['accepted'])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output, ThreadPoolExecutor(args.workers) as executor:
        write_json(args.output.with_suffix('.settings.json'), dict(
            sources={str(path): file_sha256(path) for path in [*args.candidates, args.leads]},
            fragment_prompt=PROMPT, lead_prompt=LEAD_PROMPT, workers=args.workers,
            reviewer='Qwen3.8-27B UD-IQ4_XS'))
        for verdict in executor.map(check, candidates):
            print(json.dumps(verdict, ensure_ascii=False), file=output, flush=True)


if __name__ == '__main__':
    main()
