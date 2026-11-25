"""W&B metrics and reproducible launch snapshots, excluding credentials."""
import datetime as dt
import importlib.metadata
import json
import os
import shlex
import sys
from pathlib import Path

_SECRET_KEYS = {'token','api_key','wandb_api_key','hf_token','hub_token','access_token',
                'password','secret','credential','credentials','private_key','token_file'}


def sanitized(value):
    if isinstance(value, dict):
        return {key:('[redacted]' if key.lower().replace('-','_') in _SECRET_KEYS
                     or key.lower().endswith(('_api_key','_password','_secret')) else sanitized(item))
                for key,item in value.items()}
    if isinstance(value,(list,tuple)):
        return [sanitized(item) for item in value]
    return value


def safe_argv(argv):
    result, redact_next = [], False
    for argument in argv:
        if redact_next:
            result.append('[redacted]')
            redact_next = False
            continue
        flag, separator, value = argument.partition('=')
        sensitive = flag.lstrip('-').lower().replace('-','_') in _SECRET_KEYS
        if sensitive and separator:
            result.append(flag+'=[redacted]')
        else:
            result.append(argument)
            redact_next = sensitive
    return result


def launch_parameters(config, job):
    output = Path(config['output'])
    def optional_json(path):
        return json.loads(path.read_text()) if path.is_file() else None
    command = safe_argv(getattr(sys,'orig_argv',[sys.executable,*sys.argv]))
    return sanitized(dict(
        captured_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(), job_type=job,
        command=command, shell_command=shlex.join(command), working_directory=str(Path.cwd()),
        effective_config=config, python=sys.version,
        dependencies={d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
        model_provenance=optional_json(output/'provenance.json'),
        data_manifest=optional_json(Path(config.get('data','data/prepared'))/'report.json'),
        fixed_training_parameters=dict(quantization='NF4 4-bit',double_quantization=True,
            compute_dtype='bfloat16',lora_r=16,lora_alpha=32,lora_dropout=.05,
            target_modules='language linear layers excluding lm_head',packing=False,
            thinking=False,assistant_only_loss=True,optimizer='adamw_bnb_8bit',
            epochs=1 if job == 'smoke' else config.get('epochs', 1),warmup_ratio=.03,scheduler='cosine',gradient_checkpointing=True)))


def upload_parameters(run, config, name, values):
    import wandb
    folder = Path(config['output'])/'launches'/run.id
    folder.mkdir(parents=True,exist_ok=True)
    path = folder/(name+'.json')
    path.write_text(json.dumps(sanitized(values),ensure_ascii=False,indent=2,default=str)+'\n')
    artifact = wandb.Artifact(name=f'{name}-{run.id}',type='run-configuration')
    artifact.add_file(str(path),name=path.name)
    run.log_artifact(artifact)
    return path


def save_training_arguments(run, config, arguments):
    if run is None:
        return
    parameters = launch_parameters(config,'train' if arguments.max_steps != 20 else 'smoke')
    parameters['training_arguments'] = sanitized(arguments.to_dict())
    upload_parameters(run,config,'training-parameters',parameters)
    run.config.update({'training_parameters':parameters['training_arguments']},allow_val_change=True)


def setup(config, job):
    settings = config.get('wandb', {})
    if not settings.get('enabled', True):
        return None
    import wandb
    mode = os.environ.get('WANDB_MODE', settings.get('mode', 'online'))
    if mode == 'online' and not os.environ.get('WANDB_API_KEY'):
        token_path = Path(settings.get('token_file', '.wandb/token')).expanduser()
        if not token_path.is_file() and str(token_path) == '.wandb/token':
            token_path = Path.home()/'.wandb/token'
        if not token_path.is_file():
            raise RuntimeError('W&B token missing: provide .wandb/token or WANDB_API_KEY, or set WANDB_MODE=offline')
        token = token_path.read_text().strip()
        if not token:
            raise RuntimeError('W&B token file is empty')
        os.environ['WANDB_API_KEY'] = token
    os.environ['WANDB_DISABLE_CODE'] = 'true'
    os.environ['WANDB_CONSOLE'] = 'off'
    public = sanitized({k:v for k,v in config.items() if k != 'wandb'})
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    run = wandb.init(project=settings.get('project', 'novaya-qwen35'), entity=settings.get('entity'),
                      name=settings.get('name'), job_type=job, config=public,
                      mode=mode, dir=str(output), save_code=False)
    parameters = launch_parameters(config,job)
    run.config.update({'launch_parameters':parameters},allow_val_change=True)
    upload_parameters(run,config,'launch-parameters',parameters)
    return run
