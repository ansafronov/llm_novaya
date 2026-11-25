import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import re

from novaya.data import read_rows, write_rows, write_json, file_sha256, parse_article
from novaya.editorial import request
from novaya.editorial_pair_review import unique_rows

ROOT = Path(__file__).parent
BASE = ROOT.parent
OLD = BASE / 'qwen38-style-mining-expanded-600'
TASK = ('Напиши для «Новой газеты» самостоятельный журналистский комментарий объёмом '
        '90–110 слов. Дай только текст комментария, без заголовка и предисловия. '
        'Факты бери из карточки, сохрани атрибуцию и степень уверенности.')
REWRITE = ('Перепиши карточку в компактную рабочую записку редактору на обычном русском. '
           'Сохрани все факты, имена, числа, атрибуцию, степень уверенности и авторское '
           'сопоставление, если есть. Не добавляй фактов. Избегай повторяющихся вводных '
           '«автор утверждает». Не пиши готовую статью. Верни JSON {"card":"..."}.')

def freeze():
    parts = [BASE/'qwen38-slots-benchmark/slots-2/candidates.jsonl',
             BASE/'qwen38-style-mining-no-overlap-remaining/candidates.jsonl', OLD/'candidates.jsonl']
    sources = unique_rows(parts)
    sample = unique_rows([BASE/'qwen38-style-mining-120/sample.jsonl', OLD/'sample.jsonl'])
    rows = {'train': [], 'validation': []}
    for r in read_rows(OLD/'paired-review.jsonl'):
        if not r['accepted']:
            continue
        row = sources[r['article_id']]
        if parse_article((Path('data/archive')/row['source']).read_text(), row['source'])['genre'] == 'Комментарий':
            rows[sample[r['article_id']]['split']].append(row)
    assert len(rows['validation']) == 12
    rng = random.Random(381635)
    examples = [r for r in rows['train'] if 80 <= len(r['target'].split()) <= 140]
    rng.shuffle(examples)
    assert len(examples) >= 12
    write_rows(ROOT/'sample.jsonl', [dict(id=i, article_id=r['article_id'], reference=r['target'],
        facts=r['facts'], exemplar_id=e['article_id'], exemplar=e['target'])
        for i,(r,e) in enumerate(zip(rows['validation'],examples))])

def card(value):
    result = value['card'].strip()
    if not result:
        raise ValueError('Empty card')
    return result

def rewrite():
    rows = read_rows(ROOT/'sample.jsonl')
    with ThreadPoolExecutor(2) as pool:
        write_rows(ROOT/'rewritten.jsonl', list(pool.map(
            lambda r: dict(id=r['id'], card=request(8083, REWRITE, r['facts'], card)), rows)))

def generate():
    import torch
    from novaya.model import load
    rows = read_rows(ROOT/'sample.jsonl')
    rewritten = {r['id']:r['card'] for r in read_rows(ROOT/'rewritten.jsonl')}
    config = json.loads(Path('configs/default.json').read_text())
    config['output'] = str(ROOT/'base')
    model, tok = load(config)
    tok.padding_side = 'left'
    model.eval()
    records = []
    rng = random.Random(381635)
    conditions = [(False,False),(True,False),(False,True),(True,True)]
    with (ROOT/'blind.jsonl').open('x') as out, (ROOT/'key.jsonl').open('x') as key:
        for row in rows:
            shuffled = rng.sample(conditions,len(conditions))
            prompts = []
            for edited, example in shuffled:
                facts = rewritten[row['id']] if edited else '\n'.join('- '+f for f in row['facts'])
                prompt = TASK
                if example:
                    prompt += ('\n\nОбразец интонации и композиции на другую тему. Не переноси из него '
                               'имена, события и утверждения.\n'+row['exemplar'])
                prompts.append([dict(role='user',content=prompt+'\n\nКарточка фактов\n'+facts)])
            inputs = tok.apply_chat_template(prompts, tokenize=True, return_dict=True,
                add_generation_prompt=True, enable_thinking=False, padding=True,
                return_tensors='pt').to(model.device)
            with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16):
                output = model.generate(**inputs, do_sample=False, max_new_tokens=240,
                                        use_cache=True, pad_token_id=tok.pad_token_id)
            answers = {str(i): tok.decode(ids[inputs['input_ids'].shape[1]:],
                       skip_special_tokens=True).strip() for i,ids in enumerate(output)}
            print(json.dumps(dict(id=row['id'], facts=row['facts'], reference=row['reference'],
                 answers=answers),ensure_ascii=False),file=out,flush=True)
            print(json.dumps(dict(id=row['id'], conditions={str(i):f'{int(a)}{int(b)}'
                       for i,(a,b) in enumerate(shuffled)})),file=key,flush=True)
            print(f"Generated card {row['id'] + 1}/{len(rows)}", flush=True)

JUDGE = """Сравни один журналистский комментарий с авторским образцом на ту же тему.
Оцени отдельно. voice 0–2: сходство с авторским голосом, включая ритм, выбор
деталей, композицию и интонацию. Простая связность сама по себе балла здесь не даёт.
coherence 0–2: читаемость, связность и завершённость. Оценки независимы.
missing: список номеров фактов из карточки, которые не отражены даже косвенно
(нумерация с 1). invented: список конкретных новых или искажённых утверждений.
Не считай художественные формулировки домыслами, если они не меняют фактов.
Верни JSON {"voice":0,"coherence":0,"missing":[],"invented":[],"reason":"..."}."""

def verdict(x, count):
    if any(type(x[k]) is not int or x[k] not in range(3) for k in ('voice','coherence')):
        raise ValueError('Invalid score')
    if (not isinstance(x['missing'],list) or any(type(i) is not int or not 1 <= i <= count for i in x['missing'])
        or not isinstance(x['invented'],list) or any(not isinstance(i,str) for i in x['invented'])
        or not isinstance(x['reason'],str)):
        raise ValueError('Invalid evidence')
    return dict(x, missing=sorted(set(x['missing'])))

def judge():
    rows = read_rows(ROOT/'blind.jsonl')
    def rate(job):
        row,label,answer=job
        facts=dict(enumerate(row['facts'],1))
        v=request(8084,JUDGE,dict(facts=facts,reference=row['reference'],answer=answer),
                  lambda x:verdict(x,len(facts)))
        return dict(id=row['id'],label=label,words=len(answer.split()),**v)
    with (ROOT/'scores.jsonl').open('x') as file, ThreadPoolExecutor(2) as pool:
        for row in pool.map(rate,[(r,k,v) for r in rows for k,v in r['answers'].items()]):
            print(json.dumps(row,ensure_ascii=False),file=file,flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('freeze','rewrite','generate','judge'))
    globals()[parser.parse_args().stage]()

if __name__=='__main__':
    main()
