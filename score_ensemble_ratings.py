"""Aggregate independent clip ratings; never substitutes proxy scores for taste."""
import argparse,json,random
from pathlib import Path

def score(root,files):
    key={i['id']:i for i in json.loads((root/'panel_key.json').read_text())};rows=[]
    for path in files:
        ballot=json.loads(Path(path).read_text());ratings=ballot['ratings']
        if ballot.get('panel_sha256')!=json.loads((root/'panel_identity.json').read_text())['sha256']:raise ValueError('Ballot belongs to another rendering/model panel')
        if set(ratings)!=set(key):raise ValueError('Each listener must rate all clips')
        totals={m:{k:[] for k in ['enjoyment','development','ensemble_fit']} for m in ['baseline','planned','revised']}
        for clip,values in ratings.items():
            for k in totals['baseline']:
                value=values[k]
                if not isinstance(value,(int,float)) or not 1<=value<=5:raise ValueError('Ratings must be 1..5')
                totals[key[clip]['mode']][k].append(value)
        rows.append({m:{k:sum(v)/len(v) for k,v in dims.items()} for m,dims in totals.items()})
    means={m:{k:sum(r[m][k] for r in rows)/len(rows) for k in ['enjoyment','development','ensemble_fit']} for m in ['baseline','planned','revised']}
    quality=len(rows)>=3 and all(means['revised'][k]-means['baseline'][k]>=.5 for k in ['enjoyment','ensemble_fit']) and means['revised']['development']>=means['baseline']['development']
    probes=json.loads((root/'response_probes.json').read_text()) if (root/'response_probes.json').exists() else []
    response={}
    for role in ['organ','strings','cello']:
        cases=[p for p in probes if p['role']==role]
        differences={k:sum(p['scores']['correct']-p['scores'][k] for p in cases)/max(1,len(cases)) for k in ['missing','mismatched']}
        baseline_change=sum(p['scores']['correct']-p.get('baseline_correct',float('inf')) for p in cases)/max(1,len(cases))
        intervals={};rng=random.Random(9619)
        for k in differences:
            values=[p['scores']['correct']-p['scores'][k] for p in cases]
            boots=sorted(sum(rng.choices(values,k=len(values)))/len(values) for _ in range(1000)) if values else [0.]*1000
            intervals[k]=[boots[25],boots[974]]
        response[role]=dict(compositions=len({p.get('group',p['song']) for p in cases}),mean_advantage=differences,
            change_from_baseline=baseline_change,
            bootstrap_95_intervals=intervals,
            passed=len({p.get('group',p['song']) for p in cases})>=10 and baseline_change>=0 and all(v>0 for v in differences.values()))
    manifest=json.loads((root/'data_manifest.json').read_text())
    acceptance=dict(listening=quality,partner_response=all(v['passed'] for v in response.values()),data=manifest['gate_passed'])
    (root/'acceptance.json').write_text(json.dumps(dict(passed=all(acceptance.values()),gates=acceptance),indent=2))
    result=dict(listeners=len(rows),means=means,listening_gate_passed=quality,partner_response=response,interpretation='Exploratory project threshold; not statistical proof or verification of independent raters.')
    (root/'listening_results.json').write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('ballots',nargs='+');a=p.parse_args();print(json.dumps(score(a.output,a.ballots),indent=2))
