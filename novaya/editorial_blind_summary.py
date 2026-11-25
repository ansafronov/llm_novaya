"""Join frozen blind verdicts with model identities and summarize paired results."""
import argparse
from collections import Counter
import json
from pathlib import Path

from .data import file_sha256, read_rows, write_json


def summarize(tasks, key, scores):
    by_id = {row['id']: row['models'] for row in key}
    verdicts = {(row['id'], row['label']): row for row in scores}
    expected = {(task['id'], label) for task in tasks for label in task['answers']}
    if (len(verdicts) != len(scores) or expected != verdicts.keys()
            or len(by_id) != len(key) or by_id.keys() != {task['id'] for task in tasks}):
        raise ValueError('Blind tasks, model key and verdicts must match exactly')
    outcomes, metrics = {}, {}
    for task in tasks:
        task_id, kind = task['id'], task['kind']
        if by_id[task_id].keys() != task['answers'].keys() or 'base' not in by_id[task_id].values():
            raise ValueError(f'Invalid model key for task {task_id}')
        dimensions = ('style', 'facts') if kind == 'editorial' else ('solved',)
        dimensions += ('repetition',)
        for label, model in by_id[task_id].items():
            verdict = verdicts[task_id, label]
            if verdict['kind'] != kind:
                raise ValueError(f'Verdict kind mismatch for task {task_id}')
            totals = metrics.setdefault(model, {}).setdefault(kind, Counter())
            if 'review_error' in verdict:
                totals['review_errors'] += 1
                continue
            totals['rated'] += 1
            for dimension in dimensions:
                totals[dimension] += verdict[dimension]
                for other_label, other_model in by_id[task_id].items():
                    if model == other_model:
                        continue
                    reference = verdicts[task_id, other_label]
                    if 'review_error' in reference:
                        continue
                    delta = verdict[dimension] - reference[dimension]
                    if dimension == 'repetition':
                        delta = -delta
                    comparison = (outcomes.setdefault(model, {}).setdefault(other_model, {})
                                  .setdefault(kind, {}).setdefault(dimension, Counter()))
                    comparison['win' if delta > 0 else 'loss' if delta < 0 else 'tie'] += 1
    return dict(metrics=metrics, paired_comparisons=outcomes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('blind', 'key', 'scores', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    settings = json.loads(args.scores.with_suffix('.settings.json').read_text())
    if settings['blind_sha256'] != file_sha256(args.blind):
        parser.error('Scores do not match the blind task file')
    result = summarize(read_rows(args.blind), json.loads(args.key.read_text()),
                       read_rows(args.scores))
    result['inputs'] = {str(path): file_sha256(path) for path in (args.blind, args.key, args.scores)}
    write_json(args.output, result)


if __name__ == '__main__':
    main()
