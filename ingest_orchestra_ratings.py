"""Import actual listener evidence without promoting a failed reward."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('ratings',type=Path);args=p.parse_args()
    raw=args.ratings.read_bytes();payload=json.loads(raw)
    if payload.get('schema')!='orchestra-ratings-v3':raise ValueError('Unexpected rating schema')
    audit=ROOT/'outputs/orchestra_reward_audit_v3'
    protocol=json.loads((audit/'protocol.json').read_text())
    results={r['id']:r for r in json.loads((audit/'results.json').read_text())}
    entries=[];seen=set()
    for row in payload['rows']:
        key=row['id']
        if key not in results or key in seen:raise ValueError('Unknown or duplicate clip')
        seen.add(key);ratings=row['ratings']
        if not ratings or any(k not in ['enjoyment','fit','development','warmth'] or type(v)!=int or not 1<=v<=5 for k,v in ratings.items()):raise ValueError('Invalid ratings')
        index=list(protocol['source_hashes']).index(key)
        audio=audit/f'clip_{index}.mp3'
        digest=hashlib.sha256(audio.read_bytes()).hexdigest()
        if digest!=protocol['source_hashes'][key]:raise ValueError('Listening audio changed')
        seed=9919101 if key in ['laptop','a100_128','a100_224'] else 9919101+int(key.rsplit('_',1)[1])
        entries.append(dict(id=key,ratings=ratings,audio_sha256=digest,seed=seed,group=f'seed:{seed}',
            annotation_source='user_submitted_file',reward_score=results[key]['scores']['original']))
    comparisons={}
    for category in ['enjoyment','fit','development','warmth']:
        pairs=[]
        for a,b in itertools.combinations(entries,2):
            if category not in a['ratings'] or category not in b['ratings']:continue
            if a['ratings'][category]==b['ratings'][category]:continue
            win,lose=(a,b) if a['ratings'][category]>b['ratings'][category] else (b,a)
            pairs.append(dict(preferred=win['id'],other=lose['id'],reward_agrees=win['reward_score']>lose['reward_score'],
                matched_seed=win['seed']==lose['seed']))
        comparisons[category]=dict(pairs=pairs,agreements=sum(r['reward_agrees'] for r in pairs),
            comparisons=len(pairs),note='Derived pairs share clips; they are not independent labels or a held-out test.')
    out=ROOT/'outputs/listener_preferences_v3';out.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(raw).hexdigest()
    (out/(digest+'.original.json')).write_bytes(raw)
    report=dict(schema='orchestra-preference-import-v1',source_sha256=digest,clips=entries,comparisons=comparisons,
        distinct_seed_groups=len({r['group'] for r in entries}),reward_validation_passed=False,rl_training_allowed=False,
        listener_preferred_checkpoint='outputs/a100_phrase_snapshot_v1/checkpoints/phrases.pt',
        interpretation='Step 128 is preferred on these observed clips. No conclusion about unseen step 280 or general musical quality.',
        next_evidence='Keep all versions of a seed together when splitting; acquire new seed groups for independent validation.')
    (out/'analysis.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['distinct_seed_groups','reward_validation_passed','listener_preferred_checkpoint']}))
    print(json.dumps({k:dict(agreements=v['agreements'],comparisons=v['comparisons']) for k,v in comparisons.items()}))

if __name__=='__main__':main()
