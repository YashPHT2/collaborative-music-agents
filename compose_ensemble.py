"""Compose new direct-note trio performances from a saved ensemble checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import note_trio as nt
from supervised_ensemble import OUT,load_bundle,rollout


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=OUT/'latest.pt')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=9601002)
    p.add_argument('--bars',type=int,default=24);p.add_argument('--bpm',type=int,default=100)
    args=p.parse_args();brief=nt.Brief(args.bars,args.bpm)
    if args.output.exists() and any(args.output.iterdir()):raise ValueError('Choose an empty output directory')
    args.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    model,adapters,agents,critic=load_bundle(args.checkpoint)
    ep=rollout(model,adapters,agents,critic,brief,args.seed)
    checks=nt.export(args.output,'generated',ep)
    result=dict(checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                seed=args.seed,brief=nt.asdict(brief),every_note_is_an_action=len(ep['notes'])==len(ep['decisions']),
                notes_per_role={r:sum(n[3]==s['channel'] for n in ep['notes']) for r,s in nt.ROLES.items()},audio=checks)
    (args.output/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
