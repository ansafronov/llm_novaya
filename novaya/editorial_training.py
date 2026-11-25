"""Build a training config after manual approval of a variant's pilot."""
import argparse
import json
from pathlib import Path

from .data import evenly, file_sha256, read_rows, write_json, write_rows


def editorial_rows(directory, variant, paired):
    if not paired:
        return read_rows(directory / (variant + '.jsonl'))
    pairs = read_rows(directory / 'pairs.jsonl')
    return [dict(article_id=pair['article_id'], ids=pair['ids'], **pair[variant])
            for pair in pairs if pair['fragment'] is not None and pair['lead'] is not None]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('variant', choices=('fragment', 'lead'))
    for name in ('pilot', 'train', 'validation', 'dialogues', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--dialogue-license', required=True)
    parser.add_argument('--dialogue-source', required=True)
    parser.add_argument('--approved-pairs', type=int, choices=range(24, 31), required=True)
    parser.add_argument('--incorrect-removals', type=int, choices=(0, 1), required=True)
    parser.add_argument('--paired', action='store_true', help='Use the same source fragments for both variants')
    args = parser.parse_args()
    pilot = json.loads((args.pilot / 'report.json').read_text())
    review = read_rows(args.pilot / 'review.jsonl')
    if len(review) != 30 or len({pair['article_id'] for pair in review}) < 15:
        parser.error('Pilot needs 30 reviewed examples from at least 15 articles')
    if sum(pair[args.variant] is not None for pair in review) < args.approved_pairs:
        parser.error('More approved examples claimed than available for this variant')
    config = json.loads(Path('configs/default.json').read_text())
    data = args.output / 'data'
    train = editorial_rows(args.train, args.variant, args.paired)
    val = editorial_rows(args.validation, args.variant, args.paired)
    dialogue = read_rows(args.dialogues)
    manifest_path = args.dialogues.with_suffix('.manifest.json')
    dialogue_manifest = None
    if manifest_path.exists():
        dialogue_manifest = json.loads(manifest_path.read_text())
        if (dialogue_manifest.get('exported_split') != 'train'
                or dialogue_manifest.get('exported_count') != len(dialogue)
                or dialogue_manifest.get('export_sha256') != file_sha256(args.dialogues)):
            parser.error('Dialogue export does not match its training manifest')
    if len(train) < 3200 or len(val) < 200 or len(dialogue) < 800:
        parser.error('Need 3200 train, 200 validation, and 800 approved dialogue pairs')
    if {row['article_id'] for row in train} & {row['article_id'] for row in val}:
        parser.error('Training and validation articles overlap')
    train, val, dialogue = evenly(train, 3200), evenly(val, 200), evenly(dialogue, 800)
    for row in train + val + dialogue:
        messages = row['messages']
        roles = [message['role'] for message in messages]
        if roles[:1] == ['system']:
            roles = roles[1:]
        if (not roles or roles != ['user', 'assistant'] * (len(roles) // 2)
                or any(not message['content'].strip() for message in messages)):
            parser.error('Invalid chat example')
    data.mkdir(parents=True)
    for name, rows in [('train', train + dialogue), ('validation', val), ('intermediate', val)]:
        write_rows(data / (name + '.jsonl'), rows)
    hashes = {name: file_sha256(data / (name + '.jsonl'))
              for name in ('train', 'validation', 'intermediate')}
    write_json(data / 'report.json', dict(source_revision=config['source_revision'], split_sha256=hashes))
    config.update(data=str(data), output=str(args.output / 'run'), learning_rate=3e-5,
                  eval_steps=100, recovery_save_steps=100, editorial=dict(
                      variant=args.variant, paired=args.paired,
                      counts=dict(editorial=3200, dialogues=800, validation=200),
                      hashes=hashes, dialogue_license=args.dialogue_license, dialogue_source=args.dialogue_source,
                      dialogue_manifest=dialogue_manifest,
                      labeler=pilot.get('labeler', 'Qwen/Qwen3-14B-GGUF Q4_K_M'),
                      prompts=pilot['prompts'],
                      approved_pairs=args.approved_pairs, incorrect_removals=args.incorrect_removals))
    write_json(args.output / 'config.json', config)


if __name__ == '__main__':
    main()
