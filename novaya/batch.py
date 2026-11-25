"""Optional bounded microbatch preflight; never retries the full training epoch."""
import gc
import json
import time
from pathlib import Path
from .data import read_rows, encode, write_json, experiment_identity


def accumulation_for(micro_batch, effective_batch):
    if micro_batch < 1 or effective_batch < 1 or effective_batch % micro_batch:
        raise ValueError('Microbatch must divide the effective batch size exactly')
    return effective_batch//micro_batch


def probe(config, starting_batch=2):
    import torch
    import bitsandbytes as bnb
    from accelerate.utils import find_executable_batch_size
    from .model import load, Collator, score
    if not torch.cuda.is_available():
        raise RuntimeError('Batch preflight requires CUDA access outside the sandbox')
    effective = config.get('micro_batch_size', 1)*config['gradient_accumulation']
    if starting_batch & (starting_batch-1):
        raise ValueError('Starting microbatch must be a power of two')
    accumulation_for(starting_batch, effective)
    # CPU article data is shared; all CUDA objects are created inside each attempt.
    rows = sorted(read_rows(Path(config['data'])/'train.jsonl'), key=lambda r:len(r['text']), reverse=True)[:effective]
    if len(rows) < effective:
        raise ValueError('Not enough source articles for the preflight')
    attempts = []
    candidate = [starting_batch]

    def halve():
        candidate[0] //= 2
        return candidate[0]

    @find_executable_batch_size(starting_batch_size=starting_batch, reduce_batch_size_fn=halve)
    def attempt(micro_batch):
        gc.collect()
        torch.cuda.empty_cache()
        print(f'Batch preflight: microbatch={micro_batch}, accumulation={effective//micro_batch}', flush=True)
        probe_config = dict(config, output=str(Path(config['output'])/'batch-probe'))
        model, tokenizer = load(probe_config, training=True)
        encoded = [encode(tokenizer, row, config['context']) for row in rows]
        if any(len(row['input_ids']) != config['context'] for row in encoded):
            raise ValueError('Preflight requires full-context original articles')
        loader = torch.utils.data.DataLoader(encoded, batch_size=micro_batch, collate_fn=Collator(tokenizer))
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = bnb.optim.AdamW8bit(parameters, lr=config['learning_rate'])
        model.train()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        try:
            for batch in loader:
                batch = {key:value.to(model.device) for key,value in batch.items()}
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = model(**batch).loss/accumulation_for(micro_batch,effective)
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite loss during preflight')
                loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            training_seconds = time.monotonic()-start
            evaluation = score(model, tokenizer, rows[:1], config['context'])
            result = dict(micro_batch_size=micro_batch, gradient_accumulation=effective//micro_batch,
                          effective_batch_size=effective, peak_vram_bytes=torch.cuda.max_memory_allocated(),
                          training_seconds=training_seconds, evaluation_nll=evaluation['nll'], success=True)
            attempts.append(result)
            return result
        except torch.cuda.OutOfMemoryError as exc:
            attempts.append(dict(micro_batch_size=micro_batch, success=False, reason='CUDA OOM'))
            write_json(Path(config['output'])/'batch-probe-attempts.json', attempts)
            # Accelerate recognizes this RuntimeError and retries only this preflight.
            raise RuntimeError('CUDA out of memory.') from exc

    try:
        result = attempt()
    except RuntimeError as exc:
        if 'No executable batch size found' in str(exc):
            raise torch.cuda.OutOfMemoryError('Preflight failed even at microbatch 1; stop and diagnose') from exc
        raise
    finally:
        gc.collect()
        torch.cuda.empty_cache()
    result = dict(result, attempts=attempts, identity=experiment_identity(config),
                  qualification='One optimizer step on longest articles; 20-step smoke required for a new microbatch')
    write_json(Path(config['output'])/'batch-plan.json', result)
    return result
