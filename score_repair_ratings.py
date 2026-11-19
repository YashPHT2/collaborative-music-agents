"""Keep listener preference separate from technical cooperation results."""
import argparse
import hashlib
import json
from pathlib import Path
from train_ensemble_repair import OUT,write_json


def score(root,files):
    identity=json.loads((root/'panel_identity.json').read_text())['sha256']
    dimensions=['enjoyment','development','ensemble_fit','warmth'];ids=['baseline','repaired','warm_build']
    ballots=[];seen=set()
    for path in files:
        data=Path(path).read_bytes();digest=hashlib.sha256(data).hexdigest()
        if digest in seen:raise ValueError('Duplicate ballot file')
        seen.add(digest);ballot=json.loads(data)
        if ballot.get('panel_sha256')!=identity:raise ValueError('Ballot belongs to another panel')
        if set(ballot['ratings'])!=set(ids):raise ValueError('Rate all three pieces')
        for clip in ids:
            values=ballot['ratings'][clip]
            if set(values)!=set(dimensions) or any(type(v) not in (int,float) or not 1<=v<=5 for v in values.values()):
                raise ValueError('Each rating must be 1..5')
        ballots.append(ballot['ratings'])
    if not ballots:raise ValueError('Supply at least one completed ballot')
    means={c:{d:sum(b[c][d] for b in ballots)/len(ballots) for d in dimensions} for c in ids}
    quality={c:len(ballots)>=3 and all(means[c][d]-means['baseline'][d]>=.5 for d in ['enjoyment','ensemble_fit'])
             and means[c]['development']>=means['baseline']['development'] for c in ids[1:]}
    result=dict(listeners=len(ballots),means=means,listening_gate=quality,
                warmth_change=means['warm_build']['warmth']-means['repaired']['warmth'],
                interpretation='Exploratory three-listener minimum. Rater independence is not verified; no statistical proof.')
    write_json(root/'listening_results.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUT);parser.add_argument('ballots',nargs='+')
    args=parser.parse_args();print(json.dumps(score(args.output,args.ballots),indent=2))
