"""Run or resume the approved experiment, reusing matching completed artifacts."""
import json
import subprocess
import sys
from pathlib import Path
from novaya.data import experiment_identity


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def metric_ready(path, identity, adapter=None):
    result = read_json(path)
    return (result.get('identity') == identity and result.get('adapter') == adapter
            and result.get('split') == 'validation' and result.get('target_tokens', 0) > 0)


def generations_ready(root, identity, adapter=None):
    suffix = 'adapter' if adapter else 'base'
    settings = read_json(root/f'generation-settings-{suffix}.json')
    if settings.get('identity') != identity or settings.get('adapter') != adapter:
        return False
    try:
        rows = [json.loads(line) for line in (root/f'generations-{suffix}.jsonl').read_text().splitlines()]
        return (len(rows) == 40 and sum(r.get('kind') == 'style' for r in rows) == 20
                and sum(r.get('kind') == 'general' for r in rows) == 20
                and all(r.get('generation', '').strip() for r in rows))
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def latest_checkpoint(root):
    complete = []
    for path in (root/'checkpoints').glob('checkpoint-*'):
        if not path.name.split('-')[-1].isdigit():
            continue
        if read_json(path/'complete.json').get('step') == int(path.name.split('-')[-1]) and all((path/name).is_file() for name in ['trainer_state.json','adapter_model.safetensors',
                                                  'optimizer.pt','scheduler.pt','rng_state.pth']):
            complete.append(path)
    return max(complete, key=lambda p:int(p.name.split('-')[-1]), default=None)


def main():
    config_path = Path(sys.argv[1] if len(sys.argv)>1 else 'configs/default.json')
    config = read_json(config_path)

    def command(*args, check=True):
        print('Running:', ' '.join(args), flush=True)
        result = subprocess.run([sys.executable, '-u', '-m', 'novaya.cli', '--config', str(config_path), *args])
        if check and result.returncode:
            raise SystemExit(result.returncode)
        return result

    if not (Path(config['data'])/'report.json').exists():
        command('prepare', '--fetch')
    root = Path(config['output'])
    passed = read_json(root/'smoke.json')
    keys = ('model','revision','context','seed','source_revision')
    compatible = passed.get('reload_passed') and all(
        passed.get('identity',{}).get(key) == config.get(key) for key in keys)
    compatible = compatible and passed.get('micro_batch_size',1) == config.get('micro_batch_size',1)
    if not compatible:
        smoke = command('train', '--smoke', check=False)
        if smoke.returncode == 2 and config['context'] == 4096:
            config['context'] = 2048
            config_path = root/'effective-config.json'
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps(config, indent=2)+'\n')
            command('train', '--smoke')
        elif smoke.returncode:
            raise SystemExit(smoke.returncode)
    identity = experiment_identity(config)
    if not metric_ready(root/'baseline.json', identity):
        command('evaluate', '--output-name', 'baseline')
    else:
        print('Reusing completed baseline', flush=True)
    if not generations_ready(root, identity):
        command('generate', '--suite')
    else:
        print('Reusing 40 baseline generations', flush=True)
    adapter = str(root/'adapter')
    training = read_json(root/'training.json')
    if training.get('identity') != identity or not (root/'adapter/adapter_model.safetensors').is_file():
        manifest = root/'training-identity.json'
        previous = read_json(manifest)
        if previous and previous != identity:
            raise RuntimeError('Training data/model changed: use a new output directory')
        manifest.write_text(json.dumps(identity, indent=2)+'\n')
        checkpoint = latest_checkpoint(root)
        if checkpoint:
            command('train', '--resume', str(checkpoint))
        else:
            command('train')
    if not metric_ready(root/'adapted.json', identity, adapter):
        command('evaluate', '--adapter', adapter, '--output-name', 'adapted')
    if not generations_ready(root, identity, adapter):
        command('generate', '--adapter', adapter, '--suite')
    print('Experiment complete', flush=True)


if __name__ == '__main__':
    main()
