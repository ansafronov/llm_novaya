import json
import time
from pathlib import Path
from .data import read_rows, write_json, experiment_identity
from .model import load, tokenized, Collator, score


def run(config, smoke=False, resume=None):
    import torch
    if not torch.cuda.is_available():
        write_json(Path(config['output'])/'hardware.json', {'cuda_available':False, 'torch':torch.__version__})
        raise RuntimeError('CUDA unavailable in this process; check GPU access outside the sandbox before QLoRA')
    from transformers import Trainer, TrainingArguments, TrainerCallback
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    identity = experiment_identity(config)
    write_json(output/'run-config.json', {k:v for k,v in config.items() if k != 'wandb'})
    # Re-run smoke in a fresh process at 2048 after OOM; never reuse a partial model.
    if not smoke:
        marker = output/'smoke.json'
        if not marker.exists():
            raise RuntimeError('Run train --smoke before full training')
        passed = json.loads(marker.read_text())
        if any(passed.get('identity',{}).get(k) != identity[k]
               for k in ('model','revision','context','seed','source_revision')) or not passed['reload_passed']:
            raise RuntimeError('Smoke context does not match training context')
        if passed.get('micro_batch_size',1) != config.get('micro_batch_size',1):
            raise RuntimeError('New microbatch requires train --smoke --auto-batch before full training')
        if not (output/'baseline.json').exists():
            raise RuntimeError('Run evaluate --output-name baseline before full training')
        baseline = json.loads((output/'baseline.json').read_text())
        if baseline.get('identity') != identity or baseline.get('split') != 'validation' or baseline.get('adapter'):
            raise RuntimeError('Baseline context/source differs from training')
    from .tracking import setup, save_training_arguments
    tracking = setup(config, 'smoke' if smoke else 'train')
    model, tokenizer = load(config, training=True)
    data = Path(config['data'])
    rows = read_rows(data/'train.jsonl')
    if smoke:
        rows = sorted(rows, key=lambda r: len(r.get('text', str(r.get('messages', [])))), reverse=True)[:320]
        if not rows:
            raise ValueError('Empty training data')
        rows = (rows* (320//len(rows)+1))[:320]
    train = tokenized(tokenizer, rows, config['context'], output/('smoke_tokenization.json' if smoke else 'train_tokenization.json'))
    val_rows = read_rows(data/'intermediate.jsonl')
    val = tokenized(tokenizer, val_rows, config['context'], output/'validation_tokenization.json')

    class WeightedTrainer(Trainer):
        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix='eval', **kwargs):
            result = score(self.model, tokenizer, val_rows, config['context'])
            metrics = {metric_key_prefix+'_loss':result['nll'], metric_key_prefix+'_ppl':result['ppl'],
                       metric_key_prefix+'_peak_vram_bytes':result['peak_vram_bytes'],
                       metric_key_prefix+'_seconds':result['seconds']}
            self.log(metrics)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
            return metrics

    class EpochEnd(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            if state.global_step % int(args.eval_steps):
                control.should_evaluate = True
                control.should_save = True
            return control

    class RecoverySave(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            if not smoke:
                folder = Path(args.output_dir)/f'checkpoint-{state.global_step}'
                temporary = folder/'complete.tmp.json'
                write_json(temporary, {'step':state.global_step, 'identity':identity})
                temporary.replace(folder/'complete.json')
            return control

        def on_step_end(self, args, state, control, **kwargs):
            interval = config.get('recovery_save_steps', 50)
            if not smoke and interval > 0 and state.global_step % interval == 0:
                control.should_save = True
            return control

    import inspect
    warmup = {'warmup_ratio':.03} if 'warmup_ratio' in inspect.signature(TrainingArguments).parameters else {'warmup_steps':.03}
    args = TrainingArguments(output_dir=str(output/('smoke_checkpoints' if smoke else 'checkpoints')),
        per_device_train_batch_size=config.get('micro_batch_size',1), per_device_eval_batch_size=1,
        gradient_accumulation_steps=config['gradient_accumulation'], num_train_epochs=1 if smoke else config.get('epochs', 1),
        max_steps=20 if smoke else -1, learning_rate=config['learning_rate'], **warmup,
        lr_scheduler_type='cosine', optim='adamw_bnb_8bit', bf16=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant':False},
        eval_strategy='steps', eval_steps=20 if smoke else config['eval_steps'],
        save_strategy='steps', save_steps=20 if smoke else config['eval_steps'],
        load_best_model_at_end=True, metric_for_best_model='eval_loss', greater_is_better=False,
        save_total_limit=3, logging_steps=1 if smoke else 10, report_to='wandb' if tracking else 'none', seed=config['seed'],
        data_seed=config['seed'], remove_unused_columns=False)
    save_training_arguments(tracking, config, args)
    trainer = WeightedTrainer(model=model, args=args, train_dataset=train, eval_dataset=val,
                              data_collator=Collator(tokenizer), callbacks=[EpochEnd(), RecoverySave()])
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    trainer.train(resume_from_checkpoint=resume)
    destination = output/('smoke_adapter' if smoke else 'adapter')
    trainer.save_model(str(destination))
    tokenizer.save_pretrained(destination)
    metrics = dict(identity=identity, micro_batch_size=config.get('micro_batch_size',1),
                   gradient_accumulation=config['gradient_accumulation'], context=config['context'], seconds=time.monotonic()-start,
                   peak_vram_bytes=torch.cuda.max_memory_allocated(), steps=trainer.state.global_step,
                   steps_per_second=trainer.state.global_step/(time.monotonic()-start),
                   best_checkpoint=trainer.state.best_model_checkpoint)
    if smoke:
        # Reload weights into the same frozen base after removing the original adapter.
        from peft import PeftModel
        base = model.unload()
        reloaded = PeftModel.from_pretrained(base, destination)
        check = score(reloaded, tokenizer, val_rows[:1], config['context'])
        import math
        metrics['reload_passed'] = check['target_tokens'] > 0 and math.isfinite(check['nll'])
        if not metrics['reload_passed']:
            raise RuntimeError('Reloaded adapter produced invalid loss')
    write_json(output/('smoke.json' if smoke else 'training.json'), metrics)
    if tracking:
        tracking.log(metrics)
        tracking.finish()
