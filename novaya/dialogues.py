"""Select and review Russian OpenAssistant conversations from local exports."""
import argparse
import fcntl
import gzip
import hashlib
import json
from pathlib import Path

from .data import file_sha256, normalize, write_json, write_rows, read_rows


SOURCE = 'https://huggingface.co/datasets/OpenAssistant/oasst2'
LICENSE = 'Apache-2.0'
REVIEW_FLAGS = ('spam', 'fails_task', 'lang_mismatch', 'pii', 'not_appropriate',
                'hate_speech', 'sexual_content')


def source_rows(path):
    if path.suffix == '.parquet':
        from pyarrow.parquet import ParquetFile

        for batch in ParquetFile(path).iter_batches():
            yield from batch.to_pylist()
        return
    open_file = gzip.open if path.suffix == '.gz' else open
    with open_file(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def eligible(row):
    if (row.get('lang') != 'ru' or row.get('review_result') is not True
            or row.get('deleted') is not False or row.get('synthetic') is not False
            or not (row.get('text') or '').strip()):
        return False
    role = row.get('role')
    if role != 'prompter' and (role != 'assistant' or row.get('rank') != 0):
        return False
    labels = row.get('labels') or {}
    if isinstance(labels.get('name'), list):
        labels = dict(zip(labels['name'], labels['value']))
    return all((value.get('value') if isinstance(value, dict) else value) in (0, None)
               for name in REVIEW_FLAGS if (value := labels.get(name)) is not None)


def candidates(rows, fits):
    by_id = {row['message_id']: row for row in rows if eligible(row)}
    by_tree = {}
    for row in by_id.values():
        if row['role'] != 'assistant':
            continue
        branch, seen = [], set()
        node = row
        tree_id = row['message_tree_id']
        while node is not None and node['message_id'] not in seen and node['message_tree_id'] == tree_id:
            branch.append(node)
            seen.add(node['message_id'])
            parent = node.get('parent_id')
            if parent is None:
                break
            node = by_id.get(parent)
        if not tree_id or branch[-1].get('parent_id') is not None:
            continue
        branch.reverse()
        if [node['role'] for node in branch] != ['prompter', 'assistant'] * (len(branch) // 2):
            continue
        messages = [dict(role='user' if node['role'] == 'prompter' else 'assistant',
                         content=node['text'].strip()) for node in branch]
        message_ids = [node['message_id'] for node in branch]
        score = (len(branch), message_ids[-1])
        if (tree_id not in by_tree or score > by_tree[tree_id][0]) and fits(messages):
            by_tree[tree_id] = score, dict(id=tree_id, message_ids=message_ids, messages=messages)
    ordered = sorted((row for _, row in by_tree.values()), key=lambda row: hashlib.sha256(
        ('42:' + row['id']).encode()).digest())
    seen_prompts = set()
    for row in ordered:
        prompt = normalize(row['messages'][0]['content'])
        if prompt not in seen_prompts:
            seen_prompts.add(prompt)
            yield row


def prepare(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)

    def fits(messages):
        prefix = tokenizer.apply_chat_template(
            messages[:-1], tokenize=True, return_dict=False, add_generation_prompt=True,
            enable_thinking=False)
        target = tokenizer.encode(messages[-1]['content'], add_special_tokens=False)
        return len(prefix) + len(target) + 1 <= args.context

    if args.aya or args.dolly:
        return prepare_pairs(args, fits)
    train = list(candidates(source_rows(args.train), fits))
    train_ids = {row['id'] for row in train}
    train_prompts = {normalize(row['messages'][0]['content']) for row in train}
    validation = [row for row in candidates(source_rows(args.validation), fits)
                  if row['id'] not in train_ids
                  and normalize(row['messages'][0]['content']) not in train_prompts]
    write_bundle(args, SOURCE, train, validation, {'train': args.train, 'validation': args.validation})


def prepare_pairs(args, fits):
    path = args.aya or args.dolly
    prefix = 'aya' if args.aya else 'dolly-ru'
    train, validation = [], []
    seen = set()
    for excluded in args.exclude:
        seen.update(normalize(row['messages'][0]['content']) for row in read_rows(excluded))
    for row in source_rows(path):
        if args.aya:
            if row['language_code'] != 'rus':
                continue
            prompt, answer = row['inputs'].strip(), row['targets'].strip()
        else:
            prompt, answer = row['instruction'].strip(), row['response'].strip()
            context = row['context'].strip()
            if context:
                prompt += '\n\n' + context
        key = normalize(prompt)
        if not key or not answer or key in seen:
            continue
        messages = [dict(role='user', content=prompt), dict(role='assistant', content=answer)]
        if not fits(messages):
            continue
        seen.add(key)
        digest = hashlib.sha256(('42:' + key).encode()).hexdigest()
        candidate_id = prefix + '-' + digest
        target = validation if args.aya and int(digest[:8], 16) % 5 == 0 else train
        target.append(dict(id=candidate_id, message_ids=[candidate_id + '-user',
                                                         candidate_id + '-assistant'],
                           messages=messages))
    dataset = 'CohereLabs/aya_dataset' if args.aya else 'dim/databricks_dolly_15k_ru'
    if args.dolly:
        train.sort(key=lambda row: row['id'])
    write_bundle(args, 'https://huggingface.co/datasets/' + dataset,
                 train, validation, {'train': path, 'validation': path},
                 license_name=LICENSE if args.aya else 'CC-BY-SA-3.0')


def write_bundle(args, source, train, validation, input_paths, license_name=LICENSE):
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = dict(source=source, license=license_name, revision=args.revision,
                    context=args.context, seed=42, splits={})
    for split, rows in [('train', train), ('validation', validation)]:
        filename = args.output / f'{split}.review.jsonl'
        write_rows(filename, ({**row, 'approved': None} for row in rows))
        hashes = {'count': len(rows), 'input_sha256': file_sha256(input_paths[split])}
        manifest['splits'][split] = hashes
        write_json(args.output / f'{split}.fingerprints.json', {row['id']: fingerprint(row) for row in rows})
        hashes['fingerprints_sha256'] = file_sha256(args.output / f'{split}.fingerprints.json')
    write_json(args.output / 'manifest.json', manifest)
    print(f'Train candidates: {len(train)}; validation candidates: {len(validation)}')


def fingerprint(row):
    payload = {key: row[key] for key in ('id', 'message_ids', 'messages')}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def record_decisions(args):
    lock = args.bundle / f'{args.split}.review.lock'
    with lock.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        merge_decisions(args)


def merge_decisions(args):
    manifest = json.loads((args.bundle / 'manifest.json').read_text())
    approved_rows(args.bundle, args.split, manifest)
    source = args.bundle / f'{args.split}.review.jsonl'
    rows = read_rows(source)
    candidates = {row['id']: row for row in rows}
    decisions = {}
    for path in args.decisions:
        for row in read_rows(path):
            candidate_id = row['id']
            if candidate_id not in candidates or candidate_id in decisions:
                raise ValueError(f'Unknown or duplicate decision: {candidate_id}')
            reason = row.get('reason')
            if type(row.get('approved')) is not bool or not isinstance(reason, str) or not reason.strip():
                raise ValueError(f'Decision requires a boolean and a reason: {candidate_id}')
            candidate = candidates[candidate_id]
            if candidate['approved'] is not None and candidate['approved'] != row['approved']:
                raise ValueError(f'Decision conflicts with existing review: {candidate_id}')
            decisions[candidate_id] = row
    for candidate_id, decision in decisions.items():
        candidates[candidate_id].update(approved=decision['approved'], review_reason=decision['reason'])
    temporary = source.with_suffix('.tmp')
    write_rows(temporary, rows)
    temporary.replace(source)
    print(f'{len(decisions)} decisions recorded; {sum(row["approved"] is True for row in rows)} approved')


def approved_rows(bundle, split, manifest):
    path = bundle / f'{split}.fingerprints.json'
    if file_sha256(path) != manifest['splits'][split]['fingerprints_sha256']:
        raise ValueError('Candidate fingerprints changed')
    hashes = json.loads(path.read_text())
    rows, seen = [], set()
    for row in read_rows(bundle / f'{split}.review.jsonl'):
        if row['id'] in seen or hashes.get(row['id']) != fingerprint(row):
            raise ValueError(f"Duplicate or changed review candidate: {row['id']}")
        seen.add(row['id'])
        decision = row['approved']
        if decision is not None and type(decision) is not bool:
            raise ValueError('Use true, false, or null for approved')
        if decision is True:
            rows.append({key: row[key] for key in ('id', 'message_ids', 'messages')})
    return rows


def export(args):
    rows, manifests = [], []
    prompts = set()
    bundles = [args.bundle, *getattr(args, 'supplement', [])]
    for bundle in bundles:
        manifest = json.loads((bundle / 'manifest.json').read_text())
        selected = approved_rows(bundle, args.split, manifest)
        for row in selected:
            prompt = normalize(row['messages'][0]['content'])
            if prompt in prompts:
                raise ValueError(f"Duplicate approved dialogue prompt: {row['id']}")
            prompts.add(prompt)
        rows.extend(selected)
        manifests.append(dict(**manifest, approved_count=len(selected),
                              review_sha256=file_sha256(bundle / f'{args.split}.review.jsonl')))
    other_split = 'validation' if args.split == 'train' else 'train'
    for bundle, manifest in zip(bundles, manifests):
        for row in approved_rows(bundle, other_split, manifest):
            if normalize(row['messages'][0]['content']) in prompts:
                raise ValueError(f"Approved prompt appears in both splits: {row['id']}")
    needed = 800 if args.split == 'train' else 30
    if len(rows) < needed:
        raise ValueError(f'{args.split}: {len(rows)} verified approvals; need at least {needed}')
    metadata = args.output.with_suffix('.manifest.json')
    if args.output.exists() or metadata.exists() or metadata == args.output:
        raise ValueError('Choose a new output JSONL path')
    write_rows(args.output, rows[:needed])
    provenance = manifests[0] if len(manifests) == 1 else dict(sources=manifests)
    write_json(metadata, dict(**provenance, exported_split=args.split,
                              exported_count=needed, export_sha256=file_sha256(args.output)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    selection = commands.add_parser('prepare')
    selection.add_argument('--train', type=Path)
    selection.add_argument('--validation', type=Path)
    pair_sources = selection.add_mutually_exclusive_group()
    pair_sources.add_argument('--aya', type=Path)
    pair_sources.add_argument('--dolly', type=Path)
    selection.add_argument('--exclude', nargs='*', type=Path, default=[])
    selection.add_argument('--output', required=True, type=Path)
    selection.add_argument('--revision', required=True, help='Publisher revision of both files')
    selection.add_argument('--context', type=int, default=2048)
    selection.add_argument('--tokenizer', default='runs/qwen35/adapter')
    review = commands.add_parser('export')
    review.add_argument('--bundle', required=True, type=Path)
    review.add_argument('--supplement', action='append', type=Path, default=[])
    review.add_argument('--output', required=True, type=Path)
    review.add_argument('--split', choices=('train', 'validation'), default='train')
    record = commands.add_parser('record-decisions')
    record.add_argument('--bundle', required=True, type=Path)
    record.add_argument('--split', choices=('train', 'validation'), required=True)
    record.add_argument('decisions', nargs='+', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        pair_source = args.aya or args.dolly
        if pair_source and (args.train or args.validation):
            parser.error('Choose one pair dataset or both OASST2 splits')
        if not pair_source and (not args.train or not args.validation):
            parser.error('Provide --aya, --dolly, or both --train and --validation')
    try:
        {'prepare': prepare, 'export': export, 'record-decisions': record_decisions}[args.command](args)
    except ValueError as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
