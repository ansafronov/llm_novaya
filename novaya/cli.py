import argparse
import json
import subprocess
from pathlib import Path
from .data import prepare, read_rows, write_json, write_rows, prompt, experiment_identity


def main():
    parser = argparse.ArgumentParser(description='Novaya archive QLoRA experiment')
    parser.add_argument('--config', default='configs/default.json')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--archive', default='data/archive')
    p.add_argument('--fetch', action='store_true')
    p = sub.add_parser('train')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--auto-batch', action='store_true')
    p.add_argument('--resume', nargs='?', const=True)
    p.add_argument('--context', type=int, choices=[2048,4096])
    p = sub.add_parser('probe-batch')
    p.add_argument('--starting-batch-size', type=int, choices=[1,2,4,8,16], default=2)
    p = sub.add_parser('evaluate')
    p.add_argument('--adapter')
    p.add_argument('--output-name', default='baseline')
    p.add_argument('--intermediate', action='store_true')
    p = sub.add_parser('generate')
    p.add_argument('--adapter')
    p.add_argument('--title')
    p.add_argument('--suite', action='store_true')
    p.add_argument('--max-new-tokens', type=int, default=1024)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if getattr(args, 'context', None):
        config['context'] = args.context
    if args.command == 'prepare':
        if args.fetch:
            # Partial clone and sparse checkout request only article Markdown blobs.
            subprocess.run(['git', 'clone', '--filter=blob:none', '--no-checkout',
                            'https://github.com/novaya-gazeta-ru/archive.git', args.archive], check=True)
            subprocess.run(['git','-C',args.archive,'sparse-checkout','init','--no-cone'], check=True)
            subprocess.run(['git','-C',args.archive,'sparse-checkout','set','/articles/**/*.md'], check=True)
            subprocess.run(['git','-C',args.archive,'checkout', config['source_revision']], check=True)
        if config.get('source_revision'):
            actual = subprocess.check_output(['git','-C',args.archive,'rev-parse','HEAD'], text=True).strip()
            if actual != config['source_revision']:
                raise ValueError('Archive HEAD differs from configured source_revision')
        prepare(args.archive, config['data'])
        return
    try:
        if args.command == 'probe-batch' or (args.command == 'train' and args.auto_batch):
            from .batch import probe
            plan = probe(config, getattr(args, 'starting_batch_size', 2))
            if args.command == 'probe-batch':
                return
            config['micro_batch_size'] = plan['micro_batch_size']
            config['gradient_accumulation'] = plan['gradient_accumulation']
        if args.command == 'train':
            from .train import run
            run(config, args.smoke, args.resume)
            return
        from .model import load, score
        model, tokenizer = load(config, args.adapter)
        output = Path(config['output'])
        if args.command == 'evaluate':
            if args.output_name == 'baseline' and (args.adapter or args.intermediate):
                raise ValueError('Baseline must use the base model on full validation')
            from .tracking import setup
            tracking = setup(config, args.output_name)
            split = 'intermediate' if args.intermediate else 'validation'
            result = score(model, tokenizer, read_rows(Path(config['data'])/(split+'.jsonl')), config['context'])
            result['split'] = split
            result['identity'] = experiment_identity(config)
            result['adapter'] = args.adapter
            result['source_revision'] = json.loads((Path(config['data'])/'report.json').read_text())['source_revision']
            write_json(output/(args.output_name+'.json'), result)
            if tracking:
                tracking.log(result)
                tracking.finish()
        else:
            import torch
            from transformers import set_seed
            if args.suite:
                articles = read_rows(Path(config['data'])/'headlines.jsonl')
                tasks = [dict(kind='style', prompt=prompt(r), reference=r['text'], id=r['id']) for r in articles]
                tasks += [dict(kind='general', prompt=p, id=str(i)) for i,p in enumerate(json.loads(Path('configs/general_tasks.json').read_text()))]
            elif args.title:
                tasks = [dict(kind='style', prompt=prompt({'title':args.title}))]
            else:
                raise ValueError('Specify --title or --suite')
            write_json(output/('generation-settings-adapter.json' if args.adapter else 'generation-settings-base.json'),
                       dict(identity=experiment_identity(config), adapter=args.adapter,
                            max_new_tokens=args.max_new_tokens, temperature=.7, top_p=.8, top_k=20))
            results = []
            model.eval()
            for task in tasks:
                set_seed(config['seed'])
                ids = tokenizer.apply_chat_template([{'role':'user','content':task['prompt']}],
                    tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False, return_tensors='pt').to(model.device)
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                        max_new_tokens=min(args.max_new_tokens, config['context']-ids.shape[-1]),
                        do_sample=True, temperature=.7, top_p=.8, top_k=20,
                        pad_token_id=tokenizer.pad_token_id, use_cache=True)
                text = tokenizer.decode(generated[0,ids.shape[-1]:], skip_special_tokens=True)
                result = {**task, 'generation':text, 'adapter':args.adapter}
                words = text.split()
                grams = [' '.join(words[i:i+8]) for i in range(max(0,len(words)-7))]
                result['repeated_8gram_fraction'] = 1-len(set(grams))/len(grams) if grams else 0
                if task.get('reference'):
                    reference = ' '.join(task['reference'].split())
                    result['reference_8gram_fraction'] = sum(g in reference for g in grams)/len(grams) if grams else 0
                results.append(result)
            write_rows(output/('generations-adapter.jsonl' if args.adapter else 'generations-base.jsonl'), results)
    except Exception as exc:
        import torch
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            diagnostic = {'context':config['context'], 'error':str(exc),
                          'peak_vram_bytes':torch.cuda.max_memory_allocated(),
                          'reserved_vram_bytes':torch.cuda.max_memory_reserved()}
            write_json(Path(config['output'])/'oom.json', diagnostic)
            write_json(Path(config['output'])/f"oom-{config['context']}.json", diagnostic)
            parser.exit(2, 'CUDA OOM. Retry smoke in a fresh process with --context 2048; if it also fails, stop. Use context 2048 in config for all subsequent commands.\n')
        raise


if __name__ == '__main__':
    main()
