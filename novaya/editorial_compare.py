"""Generate anonymized comparisons from held-out editorial and dialogue examples."""
import argparse
import json
import random
from pathlib import Path

from .data import evenly, normalize, read_rows, write_json, write_rows
from .model import load


def examples(directory, dialogues):
    tasks = []
    sources = [('dialogue', dialogues)]
    if directory is not None:
        sources[:0] = [('lead', directory / 'lead.jsonl'),
                       ('fragment', directory / 'fragment.jsonl')]
    for kind, path in sources:
        rows = read_rows(path)
        count = 30 if kind == 'dialogue' else 15
        if len(rows) < count:
            raise ValueError(f'{kind}: need at least {count} held-out examples')
        for row in evenly(rows, count):
            tasks.append(dict(kind=kind, messages=row['messages'][:-1],
                              reference=row['messages'][-1]['content']))
    return tasks


def paired_examples(directory, dialogues):
    fragment, lead = [read_rows(directory / variant / 'data/validation.jsonl')
                      for variant in ('fragment', 'lead')]
    if not fragment or len(fragment) != len(lead):
        raise ValueError('Paired validation requires equal, nonempty splits')
    tasks = []
    for author, synthetic in zip(fragment, lead):
        messages = author['messages'][:-1]
        if (author['article_id'] != synthetic['article_id']
                or messages != synthetic['messages'][:-1]):
            raise ValueError('Paired validation inputs differ')
        tasks.append(dict(kind='editorial', article_id=author['article_id'],
                          messages=messages, references=dict(
                              fragment=author['messages'][-1]['content'],
                              lead=synthetic['messages'][-1]['content'])))
    return evenly(tasks, 30) + examples(None, dialogues)


def grams(text):
    words = normalize(text).split()
    return list(zip(*(words[index:] for index in range(8))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument('--validation', type=Path)
    scope.add_argument('--paired-experiment', type=Path,
                       help='Root containing the fragment and lead training datasets')
    scope.add_argument('--dialogues-only', action='store_true',
                       help='Compare only on ordinary held-out dialogue tasks')
    parser.add_argument('--dialogues', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--adapter', action='append', required=True,
                        help='Adapter path; repeat to compare multiple checkpoints')
    args = parser.parse_args()
    if args.paired_experiment:
        tasks = paired_examples(args.paired_experiment, args.dialogues)
    else:
        tasks = examples(args.validation, args.dialogues)
    adapters = [None, *args.adapter]
    config = json.loads(Path('configs/default.json').read_text())
    args.output.mkdir(parents=True)
    config['output'] = str(args.output)
    import torch

    outputs = []
    for adapter in adapters:
        model_name = adapter or 'base'
        print(f'Generating {len(tasks)} answers for {model_name}', flush=True)
        model, tokenizer = load(config, adapter)
        model.eval()
        answers = []
        for index, task in enumerate(tasks, 1):
            ids = tokenizer.apply_chat_template(
                task['messages'], tokenize=True, return_dict=False, add_generation_prompt=True,
                enable_thinking=False, return_tensors='pt').to(model.device)
            budget = config['context'] - ids.shape[-1]
            if budget < 32:
                raise ValueError('Held-out prompt leaves insufficient room to generate')
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                        do_sample=False, max_new_tokens=min(384, budget), use_cache=True,
                                        pad_token_id=tokenizer.pad_token_id)
            answer = tokenizer.decode(output[0, ids.shape[-1]:], skip_special_tokens=True).strip()
            answer_grams = grams(answer)
            prompt_grams = set(grams(task['messages'][-1]['content']))
            answers.append(dict(text=answer, repeated_8gram_fraction=(
                1 - len(set(answer_grams)) / len(answer_grams) if answer_grams else 0),
                copied_8gram_fraction=(sum(gram in prompt_grams for gram in answer_grams)
                                       / len(answer_grams) if answer_grams else 0)))
            if index % 5 == 0 or index == len(tasks):
                print(f'{model_name}: {index}/{len(tasks)} answers', flush=True)
        outputs.append(answers)
        del model, tokenizer, ids, output
        torch.cuda.empty_cache()
    rng = random.Random(42)
    review, key = [], []
    for index, task in enumerate(tasks):
        order = list(range(len(adapters)))
        rng.shuffle(order)
        labels = [str(position + 1) for position in range(len(order))]
        review.append(dict(id=index, **task,
                           answers={label: outputs[item][index] for label, item in zip(labels, order)}))
        key.append(dict(id=index, models={label: adapters[item] or 'base'
                                         for label, item in zip(labels, order)}))
    write_rows(args.output / 'review.jsonl', review)
    write_json(args.output / 'key.json', key)


if __name__ == '__main__':
    main()
