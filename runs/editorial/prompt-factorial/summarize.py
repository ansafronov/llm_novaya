from collections import Counter
from pathlib import Path
from statistics import median
from novaya.data import read_rows, write_json

root = Path(__file__).parent
tasks = {r['id']:r for r in read_rows(root/'blind.jsonl')}
keys = {r['id']:r['conditions'] for r in read_rows(root/'key.jsonl')}
scores = {(r['id'],r['label']):r for r in read_rows(root/'scores.jsonl')}
assert len(tasks)==len(keys)==12 and len(scores)==48
groups = {k:{} for k in ('00','10','01','11')}
for i,task in tasks.items():
    for label,answer in task['answers'].items():
        score=scores[i,label]
        assert score['words']==len(answer.split())
        groups[keys[i][label]][i]=score
summary = {}
for k,group in groups.items():
    rows=list(group.values())
    assert len(rows)==12
    summary[k]=dict(voice=sum(r['voice'] for r in rows),
                    coherence=sum(r['coherence'] for r in rows),
                    words_median=median(r['words'] for r in rows),
                    in_range=sum(90<=r['words']<=110 for r in rows),
                    missing=sum(len(r['missing']) for r in rows),
                    with_inventions=sum(bool(r['invented']) for r in rows),
                    inventions=sum(len(r['invented']) for r in rows))
    print(k,summary[k])
paired={}
for a,b in [('10','00'),('01','00'),('11','10'),('11','01'),('11','00')]:
    paired[f'{a}_vs_{b}']={}
    for metric in ('voice','coherence'):
        delta=[groups[a][i][metric]-groups[b][i][metric] for i in tasks]
        paired[f'{a}_vs_{b}'][metric]=dict(wins=sum(x>0 for x in delta),
                                          ties=sum(x==0 for x in delta),
                                          losses=sum(x<0 for x in delta))
print(paired)
write_json(root/'summary.json',dict(summary=summary,paired=paired))
