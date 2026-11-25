import json
import subprocess
import time
from pathlib import Path
from .data import encode, perplexity, write_json


def load(config, adapter=None, training=False):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in this process; check GPU access outside the sandbox; no model substitution permitted')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 is required')
    from huggingface_hub import model_info
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration, BitsAndBytesConfig, set_seed
    set_seed(config['seed'])
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'hardware.json', dict(cuda_available=True, torch=torch.__version__,
        gpu=torch.cuda.get_device_name(0), vram_bytes=torch.cuda.get_device_properties(0).total_memory))
    lock = output/'provenance.json'
    if lock.exists():
        old = json.loads(lock.read_text())
        if old['model'] != config['model'] or (config.get('revision') and old['revision'] != config['revision']):
            raise ValueError('Model identity differs from locked run')
        revision = old['revision']
    else:
        revision = model_info(config['model'], revision=config.get('revision') or 'main').sha
        write_json(lock, dict(model=config['model'], revision=revision))
    tokenizer = AutoTokenizer.from_pretrained(config['model'], revision=revision)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    model = Qwen3_5ForConditionalGeneration.from_pretrained(config['model'], revision=revision,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16),
        dtype=torch.bfloat16, device_map={'': 0})
    model.config.use_cache = False
    model.config.get_text_config().use_cache = False
    # Keep frozen non-quantized weights identical across baseline, training and adapter evaluation.
    from peft import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training,
        gradient_checkpointing_kwargs={'use_reentrant': False})
    if training:
        from peft import get_peft_model, LoraConfig
        import bitsandbytes as bnb
        targets = [name for name, module in model.named_modules()
                   if 'language_model.' in name and isinstance(module, (torch.nn.Linear, bnb.nn.Linear4bit))
                   and not name.endswith('lm_head')]
        if not targets:
            raise RuntimeError('No language linear layers found; inspect installed Qwen architecture')
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=.05, revision=revision,
                               target_modules=targets, task_type='CAUSAL_LM'))
        assert all('lora_' in name for name, p in model.named_parameters() if p.requires_grad)
        write_json(output/'lora_targets.json', targets)
    elif adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    tokenizer.save_pretrained(output/'tokenizer')
    (output/'requirements.lock.txt').write_text(subprocess.check_output(
        [__import__('sys').executable, '-m', 'pip', 'freeze'], text=True))
    return model, tokenizer


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, rows):
        import torch
        size = max(len(r['input_ids']) for r in rows)
        return {key: torch.tensor([r[key]+[pad]*(size-len(r[key])) for r in rows])
                for key, pad in [('input_ids', self.tokenizer.pad_token_id), ('attention_mask', 0), ('labels', -100)]}


def tokenized(tokenizer, rows, context, report=None):
    from datasets import Dataset
    def examples():
        for row in rows:
            encoded = encode(tokenizer, row, context)
            if 'messages' not in row or not encoded['truncated']:
                yield encoded
    # Arrow-backed construction avoids retaining millions of Python token objects.
    dataset = Dataset.from_generator(examples, cache_dir='.cache/tokenized')
    if not len(dataset):
        raise ValueError('No training or validation examples fit the context')
    truncated = sum(dataset['truncated'])
    if report:
        write_json(report, dict(context=context, count=len(dataset), truncated=truncated,
                               truncated_fraction=truncated/len(dataset)))
    return dataset.remove_columns('truncated')



def score(model, tokenizer, rows, context):
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm
    nll, count, truncated = 0., 0, 0
    model.eval()
    start = time.monotonic()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16,
                                                enabled=str(model.device).startswith('cuda')):
        for row in tqdm(rows, desc='Validation', unit='article', mininterval=5., disable=len(rows)<20):
            encoded = encode(tokenizer, row, context)
            if 'messages' in row and encoded['truncated']:
                continue
            truncated += int(encoded['truncated'])
            batch = Collator(tokenizer)([encoded])
            batch = {k:v.to(model.device) for k,v in batch.items()}
            labels = batch.pop('labels')[:, 1:]
            logits = model(**batch).logits[:, :-1].float()
            nll += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                                   ignore_index=-100, reduction='sum').item()
            count += labels.ne(-100).sum().item()
            del logits, labels, batch
    return dict(nll_sum=nll, target_tokens=count, nll=nll/count, ppl=perplexity(nll,count),
                articles=len(rows), truncated=truncated, truncated_fraction=truncated/len(rows),
                seconds=time.monotonic()-start, context=context,
                peak_vram_bytes=torch.cuda.max_memory_allocated())
