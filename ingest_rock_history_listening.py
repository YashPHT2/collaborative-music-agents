"""Import final-candidate listening choices without changing model selection."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument('file',type=Path);a=p.parse_args();raw=a.file.read_bytes();data=json.loads(raw)
    if data.get('schema')!='rock-history-listening-v1':raise ValueError('Unknown ballot')
    run=ROOT/'outputs/rock_history_listening_v1';mapping=json.loads((run/'organizer_mapping.json').read_text());generation=json.loads((run/'generation.json').read_text());rows=[];seen=set()
    for row in data['rows']:
        key=row['comparison'];choice=row['choice']
        if key not in mapping or key in seen or choice not in ['A','B','C','Same','Neither']:raise ValueError('Invalid or duplicate choice')
        seen.add(key)
        for v in ['A','B','C']:
            if hashlib.sha256((run/key/(v+'.mp3')).read_bytes()).hexdigest()!=generation['comparisons'][key][v]['sha256']:raise ValueError('Audio identity changed')
        rows.append(dict(comparison=key,seed=mapping[key]['seed'],choice=choice,preferred_model=mapping[key].get(choice)))
    if seen!=set(mapping):raise ValueError('All three comparisons required')
    report=dict(schema=data['schema'],rows=rows,counts={name:sum(r['preferred_model']==name for r in rows) for name in ['baseline','control_final','mixed_final']},
        same=sum(r['choice']=='Same' for r in rows),neither=sum(r['choice']=='Neither' for r in rows),model_promoted=False,reward_validated=False,
        interpretation='Exploratory three-seed listening test of final candidates that failed selection; no automatic quality certification or promotion.')
    out=run/'feedback';out.mkdir(exist_ok=True);digest=hashlib.sha256(raw).hexdigest();(out/(digest+'.original.json')).write_bytes(raw);(out/(digest+'.analysis.json')).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
