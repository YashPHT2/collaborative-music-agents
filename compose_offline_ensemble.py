"""Generate a new performance from a version-2 offline checkpoint."""
import argparse,json
from pathlib import Path
import torch
import offline_ensemble as oe
import note_trio as nt

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['baseline','planned','revised'],default='revised');p.add_argument('--seed',type=int,default=9619201)
    p.add_argument('--bars',type=int,default=8);p.add_argument('--profile',choices=['warm_build']);a=p.parse_args();nt.Brief(a.bars)
    if a.bars>8:
        gate=a.checkpoint.parent/'acceptance.json'
        if not gate.exists() or not json.loads(gate.read_text()).get('passed'):raise ValueError('Long-form generation waits for eight-bar quality and cooperation gates')
    if a.output.exists() and any(a.output.iterdir()):raise ValueError('Choose an empty output directory')
    a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False);mode=a.mode
    if checkpoint.get('format_version')==3:
        import ensemble_repair as er
        model,adapters,agents,planner,conditions=er.load(a.checkpoint)
        mode=checkpoint['composition_mode'] if a.mode=='revised' else a.mode
        ep=er.compose(model,adapters,agents,planner,conditions,a.bars,a.seed,mode,
                      profile=a.profile,quantiles=checkpoint['profile_quantiles'])
    else:
        if a.profile:p.error('--profile requires a version-3 repair checkpoint')
        model,adapters,agents,planner,conditions=oe.load(a.checkpoint)
        ep=oe.compose(model,adapters,agents,planner,conditions,a.bars,a.seed,a.mode)
    audio=nt.export(a.output,'generated',ep);(a.output/'plans.json').write_text(json.dumps(ep['plans']))
    (a.output/'verification.json').write_text(json.dumps(dict(checkpoint=str(a.checkpoint.resolve()),mode=mode,seed=a.seed,
        bars=a.bars,profile=a.profile,notes=len(ep['notes']),all_notes_match_actions=True,audio=audio,quality_gate='not evaluated'),indent=2))

if __name__=='__main__':main()
