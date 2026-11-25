"""Review style pairs through a local llama.cpp server and record decisions."""
import argparse
import json
from pathlib import Path

from .data import file_sha256, read_rows, write_json
from .editorial import request

PROMPT = """Проверь пару для обучения писать как «Новая газета» по сухим сведениям.
Оцени три признака отдельно:
style — фрагмент содержит узнаваемый авторский голос, характерные наблюдения
или выразительную аргументацию, и понятен как самостоятельное задание.
clean — в цели нет заголовков, подписей, Markdown-разметки, URL, служебных
вставок и явных артефактов вёрстки. Авторская пунктуация и цензурные *** допустимы.
facts — все существенные детали фрагмента можно восстановить по карточке.
Сохрани говорящего, условия, последовательность, степень уверенности и
отношения между событиями. Откажи при существенном пропуске, подмене или домысле.
Копирование само по себе не причина для отказа. Литературную интонацию, метафору
или ритм будущая модель может восстановить по задаче и описанию существенных связей.
Верни JSON {"style":true,"clean":true,"facts":true,"reason":"конкретное основание"}.
Укажи, что потеряно или добавлено, если facts=false."""

def validate(result):
    if any(type(result[key]) is not bool for key in ('style', 'clean', 'facts')):
        raise ValueError('expected three boolean verdicts')
    if not isinstance(result['reason'], str) or not result['reason'].strip():
        raise ValueError('expected an explanation')
    return {key: result[key] for key in ('style', 'clean', 'facts', 'reason')}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--port', type=int, default=8083)
    args = parser.parse_args()
    rows = read_rows(args.candidates)
    with args.output.open('x') as output:
        write_json(args.output.with_suffix('.settings.json'), dict(
            prompt=PROMPT, input_sha256=file_sha256(args.candidates), port=args.port))
        for row in rows:
            result = dict(position=row['position'], article_id=row['article_id'], status=row['status'])
            if row['status'] != 'candidate':
                verdict = dict(style=False, clean=False, facts=False, reason=row['status'])
            else:
                try:
                    verdict = request(args.port, PROMPT, {key: row[key] for key in
                                      ('target', 'facts', 'publication_date')}, validate)
                except ValueError as error:
                    verdict = dict(style=False, clean=False, facts=False, reason=str(error))
                    result['review_error'] = True
            result.update(verdict, accepted=all(verdict[key] for key in ('style', 'clean', 'facts')))
            print(json.dumps(result, ensure_ascii=False), file=output, flush=True)

if __name__ == '__main__':
    main()
