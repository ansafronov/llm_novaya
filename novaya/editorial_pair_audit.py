"""Independently review the fixed audit sample before joining the first verdicts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from .data import file_sha256, read_rows, write_json
from .editorial_pair_review import LEAD_PROMPT, PROMPT, review, unique_rows


def compare(sample, original, audit):
    if not sample or sample.keys() != audit.keys() or sample.keys() - original.keys():
        raise ValueError('The two reviews must cover the whole audit sample')
    totals = {}
    disagreements = []
    for kind in ('fragment', 'lead'):
        dimensions = {key: dict(agree=0, disagree=0, missing=0)
                      for key in ('style', 'clean', 'facts', 'accepted')}
        for article_id, row in sample.items():
            first = original[article_id][kind]
            second = audit[article_id][kind]
            changed = []
            for key, counts in dimensions.items():
                if key not in first or key not in second or (
                        'review_error' in first or 'review_error' in second):
                    counts['missing'] += 1
                elif first[key] == second[key]:
                    counts['agree'] += 1
                else:
                    counts['disagree'] += 1
                    changed.append(key)
            if changed:
                disagreements.append(dict(article_id=article_id, position=row['position'],
                                          variant=kind, dimensions=changed,
                                          first=first, second=second))
        totals[kind] = dimensions
    return dict(sample_size=len(sample), totals=totals, disagreements=disagreements)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--original', type=Path,
                        help='Write disagreement summary after both reviews finish')
    parser.add_argument('--port', type=int, default=8084)
    args = parser.parse_args()
    rows = read_rows(args.sample)
    manifest = json.loads(args.sample.with_suffix('.manifest.json').read_text())
    if manifest['sample_sha256'] != file_sha256(args.sample):
        parser.error('Audit sample differs from its manifest')

    def check(row):
        payload = {key: row[key] for key in ('target', 'facts', 'publication_date')}
        fragment = review(args.port, PROMPT, payload)
        lead = review(args.port, LEAD_PROMPT, dict(payload, lead=row['lead']))
        return dict(article_id=row['article_id'], position=row['position'], fragment=fragment,
                    lead=lead, accepted=fragment['accepted'] and lead['accepted'])

    with args.output.open('x') as output, ThreadPoolExecutor(2) as executor:
        write_json(args.output.with_suffix('.settings.json'), dict(
            model=str(args.model), sample_sha256=manifest['sample_sha256'],
            fragment_prompt=PROMPT, lead_prompt=LEAD_PROMPT, port=args.port))
        for row in executor.map(check, rows):
            print(json.dumps(row, ensure_ascii=False), file=output, flush=True)

    if args.original:
        write_json(args.output.with_suffix('.comparison.json'), compare(
            unique_rows([args.sample]), unique_rows([args.original]), unique_rows([args.output])))


if __name__ == '__main__':
    main()
