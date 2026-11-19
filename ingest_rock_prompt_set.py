"""Store three composition-group context choices without fitting a reward."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('file',type=Path);a=p.parse_args();raw=a.file.read_bytes();ballot=json.loads(raw)
    if ballot.get('schema')!='rock-prompt-set-v1' or ballot.get('checkpoint_step')!=128:raise ValueError('Unknown ballot')
    run=ROOT/'outputs/rock_prompt_set_v1';protocol=json.loads((run/'protocol.json').read_text());prompts={r['id']:r for r in protocol['prompts']}
    seen=set();rows=[]
    for row in ballot['rows']:
        key=row['prompt'];choice=row['choice']
        if key not in prompts or key in seen or choice not in ['prompted','unprompted','Same','Neither']:raise ValueError('Invalid/duplicate vote')
        seen.add(key);generation=json.loads((run/key/'generation.json').read_text())
        for condition in ['unprompted','prompted']:
            name='step128_'+condition;digest=hashlib.sha256((run/key/(name+'.mp3')).read_bytes()).hexdigest()
            if digest!=generation['clips'][name]['sha256']:raise ValueError('Listening audio changed')
        rows.append(dict(prompt=key,choice=choice,seed=prompts[key]['seed'],composition_group=prompts[key]['group'],source_split=prompts[key]['split']))
    if seen!=set(prompts):raise ValueError('All three votes required')
    report=dict(schema=ballot['schema'],rows=rows,counts={v:sum(r['choice']==v for r in rows) for v in ['prompted','unprompted','Same','Neither']},
        distinct_composition_groups=len({r['composition_group'] for r in rows}),reward_validated=False,new_training=False,
        interpretation='Small non-blind context diagnostic. Keep composition groups out of reward fitting; no general quality or mood conclusion.')
    out=run/'feedback';out.mkdir(exist_ok=True);digest=hashlib.sha256(raw).hexdigest();(out/(digest+'.original.json')).write_bytes(raw);(out/(digest+'.analysis.json')).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
