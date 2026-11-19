"""Decode exported JSON using the organizer-only map; keep diagnostics separate."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('files',type=Path,nargs='+')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/band_team_ballot_v2/decoded_votes.json')
    args=p.parse_args()
    decoded=[]; hashes=set()
    for file in args.files:
        digest=hashlib.sha256(file.read_bytes()).hexdigest()
        if digest in hashes: raise ValueError('Duplicate vote export supplied')
        hashes.add(digest); vote=json.loads(file.read_text(encoding='utf-8-sig'))
        ballot=vote.get('ballot_id')
        if ballot not in ['band-team-v2','band-quick-v1']: raise ValueError('Wrong ballot')
        folder='band_quick_vote' if ballot=='band-quick-v1' else 'band_team_ballot_v2'
        organizer=json.loads((ROOT/f'outputs/{folder}/organizer_manifest.json').read_text())
        mapping={r['id']:r for r in organizer['pairs']}
        seen=set()
        for row in vote['ratings']:
            pid=row['id']
            if pid not in mapping or pid in seen: raise ValueError('Unknown or duplicate comparison')
            seen.add(pid); case=mapping[pid]; result={}
            for metric in (['enjoyment'] if ballot=='band-quick-v1' else ['enjoyment','coherence','together']):
                choice=row.get('choice' if ballot=='band-quick-v1' else metric,'') or ''
                choice={'Same':'Tie','Neither':'Neither good'}.get(choice,choice)
                if choice not in ['','A','B','Tie','Neither good']: raise ValueError('Invalid rating')
                result[metric]=case['mapping'][choice]['method'] if choice in ['A','B'] else choice or 'Unrated'
            decoded.append(dict(source_file=file.name,participant=vote.get('participant',''),id=pid,
                                ballot_id=ballot,passage=case['passage'],kind=case.get('kind','reward_training'),preferences=result,comment=row.get('comment','')))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    ballot_ids=sorted({r['ballot_id'] for r in decoded})
    args.output.write_text(json.dumps(dict(ballot_id=ballot_ids[0] if len(ballot_ids)==1 else 'mixed',export_files=len(hashes),ratings=decoded,
        note='Repeated passages and multiple judgments from one person are correlated. Separate training from diagnostics.'),indent=2))
    print(f'Decoded {len(decoded)} comparison ratings from {len(hashes)} export files to {args.output}')

if __name__=='__main__': main()
