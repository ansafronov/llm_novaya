"""Freeze matched QLoRA datasets for the fragment-versus-lead experiment."""
import argparse
import json
from pathlib import Path
import random

from .data import encode, file_sha256, read_rows, write_json, write_rows
from .editorial import example
from .editorial_pair_review import unique_rows

TASK = 'Напиши текст для «Новой газеты» по этим фактам.'


def build(args):
    sources = unique_rows(args.candidates)
    sources = {key: row for key, row in sources.items() if row['status'] == 'candidate'}
    leads = unique_rows([args.leads])
    verdicts = unique_rows([args.review])
    splits = unique_rows(args.samples)
    if sources.keys() != leads.keys() or sources.keys() != verdicts.keys():
        raise ValueError('Every candidate must have a lead result and a paired verdict')
    settings_path = args.review.with_suffix('.settings.json')
    review_settings = json.loads(settings_path.read_text())
    for path in [*args.candidates, args.leads]:
        if review_settings['sources'].get(str(path)) != file_sha256(path):
            raise ValueError(f'Review does not match {path}')

    rows = {variant: {'train': [], 'validation': []} for variant in ('fragment', 'lead')}
    for article_id, source in sources.items():
        lead, verdict, sample = leads[article_id], verdicts[article_id], splits[article_id]
        if any(record['position'] != source['position'] for record in (lead, verdict, sample)):
            raise ValueError(f'Position mismatch for {article_id}')
        if sample['source'] != source['source'] or lead['source'] != source['source']:
            raise ValueError(f'Source mismatch for {article_id}')
        if not verdict['accepted']:
            continue
        if (lead['status'] != 'candidate'
                or not verdict['fragment']['accepted'] or not verdict['lead']['accepted']):
            raise ValueError(f'Inconsistent approval for {article_id}')
        card = '\n'.join('- ' + fact for fact in source['facts'])
        for variant, target in (('fragment', source['target']), ('lead', lead['lead'])):
            rows[variant][sample['split']].append(dict(article_id=article_id,
                **example(TASK, card, target)))

    from transformers import AutoTokenizer
    config = json.loads(args.config.read_text())
    tokenizer = AutoTokenizer.from_pretrained(config['model'], revision=config['revision'],
                                              local_files_only=True)

    def target_tokens(row):
        encoded = encode(tokenizer, row, config['context'])
        if encoded['truncated']:
            return 0
        return sum(label != -100 for label in encoded['labels'][1:])

    excluded_for_context = {}
    editorial_tokens = {'fragment': {}, 'lead': {}}
    for split in ('train', 'validation'):
        fragment, lead = rows['fragment'][split], rows['lead'][split]
        kept = [(a, b, na, nb) for a, b in zip(fragment, lead)
                if (na := target_tokens(a)) and (nb := target_tokens(b))]
        excluded_for_context[split] = len(fragment) - len(kept)
        for variant, row_index, token_index in (('fragment', 0, 2), ('lead', 1, 3)):
            rows[variant][split] = [item[row_index] for item in kept]
            editorial_tokens[variant][split] = sum(item[token_index] for item in kept)

    train_count = len(rows['fragment']['train'])
    validation_count = len(rows['fragment']['validation'])
    if not train_count or not validation_count:
        raise ValueError('Both splits must contain approved pairs')
    manifest_path = args.dialogues.with_suffix('.manifest.json')
    dialogue_manifest = json.loads(manifest_path.read_text())
    pool = read_rows(args.dialogues)
    if (dialogue_manifest['export_sha256'] != file_sha256(args.dialogues)
            or dialogue_manifest['exported_count'] != len(pool)
            or dialogue_manifest['exported_split'] != 'train'):
        raise ValueError('Dialogue export does not match its training manifest')
    dialogue_counts = [(row, count) for row in pool if (count := target_tokens(row))]
    dialogue_pairs = random.Random(config['seed']).sample(dialogue_counts, (train_count + 2) // 4)
    dialogue = [row for row, count in dialogue_pairs]
    dialogue_tokens = sum(count for row, count in dialogue_pairs)
    inputs = [*args.candidates, *args.samples, args.leads, args.review, settings_path, args.dialogues]
    provenance = {str(path): file_sha256(path) for path in inputs}
    for variant, splits in rows.items():
        directory = args.output / variant
        data = directory / 'data'
        data.mkdir(parents=True, exist_ok=False)
        write_rows(data / 'train.jsonl', splits['train'] + dialogue)
        for split in ('validation', 'intermediate'):
            write_rows(data / f'{split}.jsonl', splits['validation'])
        hashes = {split: file_sha256(data / f'{split}.jsonl')
                  for split in ('train', 'validation', 'intermediate')}
        write_json(data / 'report.json', dict(source_revision=config['source_revision'],
                                             split_sha256=hashes))
        write_json(directory / 'config.json', dict(
            config, data=str(data), output=str(directory / 'run'), learning_rate=3e-5,
            editorial=dict(variant=variant, paired=True, inputs=provenance, hashes=hashes,
                           counts=dict(editorial=train_count, dialogues=len(dialogue),
                                       validation=validation_count),
                           dialogue_fraction=len(dialogue) / (train_count + len(dialogue)),
                           excluded_for_context=excluded_for_context,
                           target_tokens=dict(editorial=editorial_tokens[variant],
                                              dialogues=dialogue_tokens),
                           dialogue_target_token_fraction=dialogue_tokens / (
                               dialogue_tokens + editorial_tokens[variant]['train']),
                           dialogue_manifest=dialogue_manifest)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('candidates', 'samples'):
        parser.add_argument('--' + name, type=Path, nargs='+', required=True)
    for name in ('leads', 'review', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--dialogues', type=Path, default=Path('data/dialogues.jsonl'))
    parser.add_argument('--config', type=Path, default=Path('configs/default.json'))
    args = parser.parse_args()
    build(args)


if __name__ == '__main__':
    main()
