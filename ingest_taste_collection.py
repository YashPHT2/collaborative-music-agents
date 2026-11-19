"""Store taste ratings with audio identity and composition groups preserved."""
import argparse,hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('file',type=Path);a=p.parse_args()
    raw=a.file.read_bytes();data=json.loads(raw)
    if data.get('schema')!='taste-collection-v1':raise ValueError('Unexpected schema')
    run=ROOT/'outputs/taste_collection_v1';mapping=json.loads((run/'organizer_mapping.json').read_text())
    rows=[];seen=set();categories={'enjoyment','fit','development','warmth'}
    for row in data['rows']:
        key=row['id'];ratings=row['ratings']
        if key not in mapping or key in seen:raise ValueError('Unknown or duplicate clip')
        if set(ratings)!=categories or any(type(v)!=int or not 1<=v<=5 for v in ratings.values()):raise ValueError('Four integer ratings from 1 to 5 required')
        entry=mapping[key]
        if hashlib.sha256((run/(key+'.mp3')).read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('Audio changed')
        seen.add(key);rows.append(dict(id=key,ratings=ratings,**entry))
    if seen!=set(mapping):raise ValueError('Complete all six clips')
    report=dict(schema=data['schema'],rows=rows,distinct_seed_groups=len({r['group'] for r in rows}),new_seed_groups=len({r['group'] for r in rows if not r['repeat_anchor']}),
        interpretation='Preference development data. Repeat anchors are consistency checks, not independent new samples. Group all versions of each seed when splitting.',reward_validated=False,rl_training_allowed=False)
    out=ROOT/'outputs/listener_preferences_v3/taste_development';out.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(raw).hexdigest();(out/(digest+'.original.json')).write_bytes(raw)
    (out/(digest+'.analysis.json')).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
