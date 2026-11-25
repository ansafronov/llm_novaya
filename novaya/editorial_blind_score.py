"""Score anonymized adapter answers through a local llama.cpp server."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from .data import file_sha256, read_rows, write_json
from .editorial import request

EDITORIAL = """Оцени один ответ на редакционное задание. Карточка — источник фактов.
Эталоны показывают два допустимых формата и особенности языка.
Оцени язык издания (style) и точность (facts) по шкале 0–2.
style: 0 — сухой, шаблонный или распадающийся текст; 1 — связный текст;
2 — авторская интонация, точный выбор деталей и передача оттенков смысла.
facts: 0 — серьёзная подмена или домысел; 1 — небольшая неточность;
2 — существенные утверждения подтверждены, атрибуция и оговорки сохранены.
Отметь repetition (true), если повторы мешают читать. Совпадение с карточкой
само по себе допустимо. Верни JSON {"style":2,"facts":2,"repetition":false,"reason":"..."}."""

DIALOGUE = """Оцени ответ на обычный запрос с учётом истории диалога. Эталон —
один допустимый ответ. Учитывай выполнение задачи, точность, язык и соблюдение
уточнений пользователя. solved=true, если задача решена без существенных
ошибок. repetition=true, если повторы мешают читать.
Верни JSON {"solved":true,"repetition":false,"reason":"..."}."""


def validate(value, editorial):
    keys = ('style', 'facts') if editorial else ()
    if any(type(value.get(key)) is not int or value[key] not in (0, 1, 2) for key in keys):
        raise ValueError('Expected integer scores between 0 and 2')
    flags = ('repetition',) if editorial else ('repetition', 'solved')
    if any(type(value.get(key)) is not bool for key in flags):
        raise ValueError('Expected boolean flags')
    if not isinstance(value.get('reason'), str) or not value['reason'].strip():
        raise ValueError('Expected a reason')
    return {key: value[key] for key in (*keys, *flags, 'reason')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blind', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--port', type=int, default=8084)
    args = parser.parse_args()
    tasks = read_rows(args.blind)

    def score(job):
        task, label, answer = job
        editorial = task['kind'] == 'editorial'
        fields = ('messages', 'references') if editorial else ('messages', 'reference')
        payload = {key: task[key] for key in fields}
        try:
            verdict = request(args.port, EDITORIAL if editorial else DIALOGUE,
                              dict(payload, answer=answer['text']),
                              lambda value: validate(value, editorial))
        except ValueError as error:
            verdict = dict(review_error=str(error))
        return dict(id=task['id'], label=label, kind=task['kind'], **verdict)

    jobs = [(task, label, answer) for task in tasks for label, answer in task['answers'].items()]
    with args.output.open('x') as output, ThreadPoolExecutor(2) as executor:
        write_json(args.output.with_suffix('.settings.json'), dict(
            blind_sha256=file_sha256(args.blind), editorial_prompt=EDITORIAL,
            dialogue_prompt=DIALOGUE, port=args.port))
        for row in executor.map(score, jobs):
            print(json.dumps(row, ensure_ascii=False), file=output, flush=True)


if __name__ == '__main__':
    main()
