"""Archive cleaning and chronological split, independent of model tokenization."""
import collections
import datetime as dt
import hashlib
import json
import re
import subprocess
from pathlib import Path
import yaml


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def read_rows(path):
    with open(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_rows(path, rows):
    with open(path, 'w') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def normalize(text):
    return re.sub(r'\s+', ' ', text).strip().casefold()


def parse_article(raw, source):
    # The archive contains CP1252 punctuation stored as C1 control characters.
    raw = re.sub(r'[\x80-\x9f]', lambda m: m[0].encode('latin1').decode('cp1252', errors='replace').replace('\ufffd',' '), raw)
    match = re.match(r'\A---\r?\n(.*?)\r?\n---\r?\n(.*)\Z', raw, re.S)
    if not match:
        raise ValueError('missing_frontmatter')
    try:
        header = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', ' ', match[1])
        meta = yaml.load(header, Loader=getattr(yaml, 'CSafeLoader', yaml.SafeLoader))
    except yaml.YAMLError as exc:
        raise ValueError('invalid_yaml') from exc
    if not isinstance(meta, dict):
        raise ValueError('invalid_yaml')
    title = str(meta.get('title') or '').strip()
    if not title:
        raise ValueError('missing_title')
    try:
        published = dt.datetime.fromisoformat(str(meta.get('date', '')).replace('Z', '+00:00'))
        if published.tzinfo is None:
            published = published.replace(tzinfo=dt.timezone.utc)
        published = published.astimezone(dt.timezone.utc)
        date = published.date().isoformat()
    except ValueError as exc:
        raise ValueError('invalid_date') from exc
    body = re.sub(r'!\[[^\]]*\]\([^\n]*?\)', '', match[2])
    lines = body.splitlines()
    # Remove only a repeated leading headline, never body subheadings.
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and normalize(re.sub(r'^#+\s*', '', lines[0])) == normalize(title):
        lines.pop(0)
    cleaned, related = [], False
    for line in lines:
        if re.match(r'^\s*(?:>\s*)?(?:\*\*)?Читайте также:?(?:\*\*)?\s*:?.*$', line, re.I):
            related = True
            continue
        if related and not line.strip():
            cleaned.append(line)
            continue
        if related and re.fullmatch(r'\s*(?:[-*]\s*)?(?:[^\[\n]{1,100}:\s*)?\[.*\]\(.*\)\s*', line):
            continue
        related = False
        cleaned.append(line)
    lines = cleaned
    body = '\n'.join(line for line in lines if not re.match(
        r'^\s*(?:<!--.*-->|(?:\*\*)?(?:Видео|Галерея|YouTube|Telegram|ВКонтакте):?(?:\*\*)?\s*(?:https?://\S+|\[[^\]]*\]\([^\n]*\)))\s*$', line, re.I))
    body = re.sub(r'\n{3,}', '\n\n', body).strip()
    if len(re.findall(r'\w+', body)) < 50:
        raise ValueError('short_text')
    return dict(title=title, date=date, published_at=published.isoformat(), genre=str(meta.get('genre') or ''),
                rubric=str(meta.get('rubric') or ''), url=str(meta.get('source') or ''),
                text=body, source=source, id=hashlib.sha256(normalize(body).encode()).hexdigest())


def split_rows(rows):
    rows = sorted(rows, key=lambda r: (r['date'], r.get('published_at',r['date']), r['source']))
    seen, unique = set(), []
    for row in rows:
        if row['id'] not in seen:
            seen.add(row['id'])
            unique.append(row)
    boundaries = [i for i in range(1, len(unique)) if unique[i-1]['date'] < unique[i]['date']]
    if not boundaries:
        raise ValueError('At least two distinct publication days are required')
    cut = min(boundaries, key=lambda i: (abs(len(unique)-i-len(unique)*.1), i))
    return unique[:cut], unique[cut:], len(rows)-len(unique)


def evenly(rows, n):
    if len(rows) <= n:
        return rows[:]
    return [rows[round(i*(len(rows)-1)/(n-1))] for i in range(n)]


def temporal_sample(rows, n):
    """Nearest unused articles to evenly spaced calendar dates, in date order."""
    from bisect import bisect_left
    if len(rows) <= n:
        return rows[:]
    days = [dt.date.fromisoformat(r['date']).toordinal() for r in rows]
    used = set()
    for i in range(n):
        target = days[0]+(days[-1]-days[0])*i/(n-1)
        right = bisect_left(days, target)
        left = right-1
        while True:
            if left >= 0 and (right >= len(days) or target-days[left] <= days[right]-target):
                index = left
                left -= 1
            else:
                index = right
                right += 1
            if index not in used:
                used.add(index)
                break
    return [rows[i] for i in sorted(used)]


def prepare(archive, output):
    archive, output = Path(archive), Path(output)
    revision = subprocess.check_output(['git', '-C', str(archive), 'rev-parse', 'HEAD'], text=True).strip()
    if subprocess.check_output(['git', '-C', str(archive), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Source checkout must be clean')
    rows, rejected = [], collections.Counter()
    for file in sorted((archive/'articles').rglob('*.md')):
        try:
            rows.append(parse_article(file.read_text(), str(file.relative_to(archive))))
        except (ValueError, UnicodeError) as exc:
            rejected[str(exc)] += 1
    train, val, duplicates = split_rows(rows)
    output.mkdir(parents=True, exist_ok=True)
    for name, subset in [('train', train), ('validation', val), ('intermediate', temporal_sample(val, 512)), ('headlines', temporal_sample(val, 20))]:
        write_rows(output/f'{name}.jsonl', subset)
    report = {'source_revision': revision, 'source_files':len(rows)+sum(rejected.values()),
              'accepted_before_dedup':len(rows), 'excluded': dict(rejected), 'duplicates': duplicates}
    for name, subset in [('train', train), ('validation', val)]:
        lengths = sorted(len(r['text']) for r in subset)
        report[name] = dict(count=len(subset), dates=[subset[0]['date'], subset[-1]['date']],
                            genres=dict(collections.Counter(r['genre'] for r in subset)),
                            characters={'min': lengths[0], 'median': lengths[len(lengths)//2], 'max': lengths[-1],
                                        'p10':lengths[round((len(lengths)-1)*.1)],
                                        'p90':lengths[round((len(lengths)-1)*.9)],
                                        'p99':lengths[round((len(lengths)-1)*.99)],
                                        'histogram':dict(collections.Counter(
                                            '<1000' if n<1000 else '1000-4999' if n<5000 else
                                            '5000-9999' if n<10000 else '10000-19999' if n<20000 else
                                            '20000-49999' if n<50000 else '>=50000' for n in lengths))})
    report['split_sha256'] = {name:file_sha256(output/(name+'.jsonl'))
                              for name in ('train','validation','intermediate','headlines')}
    write_json(output/'report.json', report)


def prompt(row):
    lines = ['Напиши материал в стилистике «Новой газеты».', 'Заголовок: '+row['title']]
    for field, label in [('genre', 'Жанр'), ('rubric', 'Рубрика')]:
        if row.get(field):
            lines.append(label+': '+row[field])
    return '\n'.join(lines)


def encode(tokenizer, row, context):
    messages = row['messages'][:-1] if 'messages' in row else [{'role': 'user', 'content': prompt(row)}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)
    # Encode target independently so the prefix includes the empty thinking block.
    target = tokenizer.encode(row['messages'][-1]['content'] if 'messages' in row else row['text'],
                              add_special_tokens=False)
    end = tokenizer.convert_tokens_to_ids('<|im_end|>')
    if end is None or end == tokenizer.unk_token_id:
        raise ValueError('Missing Qwen completion token')
    if len(prefix) >= context-1 and 'messages' not in row:
        raise ValueError('Prompt leaves no room for target')
    full = prefix + target + [end]
    ids = full[:context]
    return dict(input_ids=ids, attention_mask=[1]*len(ids), labels=([-100]*len(prefix)+target+[end])[:context], truncated=len(full)>context)


def perplexity(nll, count):
    import math
    if count <= 0:
        raise ValueError('No target tokens')
    return math.exp(nll/count) if nll/count < 709 else float('inf')


def experiment_identity(config):
    report = json.loads((Path(config['data'])/'report.json').read_text())
    for name, expected in report['split_sha256'].items():
        if file_sha256(Path(config['data'])/(name+'.jsonl')) != expected:
            raise ValueError(f'Prepared {name} file changed; rerun prepare before this experiment')
    return dict(model=config['model'], revision=config.get('revision'), context=config['context'],
                seed=config['seed'], source_revision=report['source_revision'],
                split_sha256=report['split_sha256'])
