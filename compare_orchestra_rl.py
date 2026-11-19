"""Matched supervised versus 19-role PPO candidate, without cherry-picking."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
import soundfile as sf
import torch
import orchestra_joint_rl as rl
import evaluate_trained_orchestra as preview
from audit_projected_roles import loudness
from train_ensemble_repair import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--rl',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=9919101)
    args=p.parse_args();out=args.output.resolve()
    if (out/'evaluation.json').exists():raise ValueError('Keep existing comparison; use a new output')
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    saved=torch.load(args.rl,map_location='cpu',weights_only=False)
    if saved.get('format_version')!=rl.FORMAT or saved['supervised_sha256']!=rl.sha(args.source):raise ValueError('RL/source mismatch')
    model,adapters,planner,conditions,neutral,source=rl.load_supervised(args.source)
    frozen=rl.band.fingerprint(model);actors={r:rl.oe.Musician().eval() for r in rl.l.ROLES};critic=rl.OrchestraCritic().eval()
    preview.OUT=out;preview.BARS=8;preview.BPM=70;preview.SECONDS=32*round(60_000_000/70)/1_000_000+3
    raws={};checks={}
    for name in ['supervised','rl_candidate']:
        if name=='rl_candidate':
            for r,a in actors.items():a.load_state_dict(saved['actors'][r])
        ep=rl.rollout(model,adapters,actors,planner,conditions,neutral,critic,8,args.seed,
            source.get('composition_mode','planned'),source.get('calm_profile'),lambda *_:0.)
        write_json(out/(name+'_episode.json'),{k:v for k,v in ep.items() if k!='steps'})
        raw,parts,fade=preview.render(name,ep);del parts,fade
        raws[name]=raw;checks[name]=dict(participation=preview.participation(ep['notes']),notes_match_actions=True)
    levels={n:loudness(path) for n,path in raws.items()}
    target=min([-22.]+[-2-(peak-integrated) for integrated,peak in levels.values() if np.isfinite(integrated)])
    for name,path in raws.items():
        gain=target-levels[name][0] if np.isfinite(levels[name][0]) else 0.
        wav=out/(name+'.wav');mp3=out/(name+'.mp3')
        subprocess.run(['ffmpeg','-y','-v','error','-i',str(path),'-af',f'volume={gain:.9f}dB','-c:a','pcm_s24le',str(wav)],check=True)
        subprocess.run(['ffmpeg','-y','-v','error','-i',str(wav),'-q:a','2',str(mp3)],check=True)
        decoded=out/(name+'_decoded.wav');subprocess.run(['ffmpeg','-y','-v','error','-i',str(mp3),str(decoded)],check=True)
        audio,sr=sf.read(decoded);decoded.unlink()
        if not np.isfinite(audio).all() or np.max(np.abs(audio))>=1:raise ValueError('Nonfinite/clipped output')
        checks[name].update(seconds=len(audio)/sr,peak=float(np.max(np.abs(audio))),lufs=loudness(wav)[0])
    if rl.band.fingerprint(model)!=frozen:raise ValueError('Frozen model changed')
    write_json(out/'evaluation.json',dict(seed=args.seed,source_sha256=rl.sha(args.source),rl_sha256=rl.sha(args.rl),
        audio=checks,same_renderer_and_profile=True,frozen_unchanged=True,listener_preference='unknown',cooperation_established=False))
    (out/'listen.html').write_text('<!doctype html><meta charset="utf-8"><title>Supervised / PPO</title><body style="max-width:850px;margin:40px auto;font:18px system-ui"><h1>Did joint RL help?</h1><p>Same seed, instruments, profile and loudness. PPO is an experimental candidate.</p>'+''.join('<h2>'+label+'</h2><audio controls style="width:100%" src="'+name+'.mp3"></audio>' for name,label in [('supervised','Supervised foundation'),('rl_candidate','Joint PPO candidate')])+'<p><a href="evaluation.json">Every-role participation and checks</a></p>',encoding='utf-8')
    print(out/'listen.html')


if __name__=='__main__':main()
