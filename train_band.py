"""Train three musicians jointly; resumeable, bounded local experiments."""
import argparse
import copy
import json
import random
import time
from pathlib import Path
import torch
from band import (ROOT,ROLES,Brief,Musician,TeamCritic,fingerprint,load_model,rollout,update,export_performance)

def make_policies(): return {r:Musician() for r in ROLES}

def save_checkpoint(path,policies,critic,optimizers,critic_optimizer,initial,completed,seed,logs,elapsed):
    payload=dict(actors={r:a.state_dict() for r,a in policies.items()},critic=critic.state_dict(),
                 optimizers={r:o.state_dict() for r,o in optimizers.items()},critic_optimizer=critic_optimizer.state_dict(),
                 initial=initial,completed=completed,seed=seed,logs=logs,elapsed=elapsed,
                 torch_rng=torch.get_rng_state(),python_rng=random.getstate(),format_version=1)
    temp=path.with_suffix('.tmp'); torch.save(payload,temp); temp.replace(path)

def generate(args):
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    policies=make_policies(); critic=TeamCritic()
    for r,a in policies.items(): a.load_state_dict(checkpoint['actors'][r])
    critic.load_state_dict(checkpoint['critic'])
    model=load_model(args.device); brief=Brief(args.bars,args.bpm,args.tonic,not args.major)
    result=rollout(model,policies,critic,brief,args.seed,shared=not args.no_partner)
    checks=export_performance(args.output,'trained',result,not args.no_audio)
    (args.output/'verification.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2),flush=True)

def train(args):
    root=args.output; root.mkdir(parents=True,exist_ok=True)
    if (root/'latest.pt').exists() and not args.resume:
        raise ValueError('Run already exists. Use --resume or a new --output directory.')
    torch.manual_seed(args.seed); random.seed(args.seed)
    policies=make_policies(); critic=TeamCritic()
    optimizers={r:torch.optim.Adam(a.parameters(),lr=1e-4) for r,a in policies.items()}
    critic_optimizer=torch.optim.Adam(critic.parameters(),lr=3e-4)
    initial={r:copy.deepcopy(a.state_dict()) for r,a in policies.items()}
    completed=0; logs=[]; elapsed=0
    if args.resume:
        payload=torch.load(root/'latest.pt',map_location='cpu',weights_only=False)
        if args.seed!=payload['seed']: raise ValueError('Resume with original seed')
        previous=json.loads((root/'settings.json').read_text())
        for key in ['episodes','bars']:
            if getattr(args,key)!=previous[key]:
                raise ValueError(f'Resume must preserve --{key}; start a new output directory to change it')
        for r,a in policies.items():
            a.load_state_dict(payload['actors'][r]); optimizers[r].load_state_dict(payload['optimizers'][r])
        critic.load_state_dict(payload['critic']); critic_optimizer.load_state_dict(payload['critic_optimizer'])
        initial=payload['initial']; completed=payload['completed']; logs=payload['logs']; elapsed=payload['elapsed']
        torch.set_rng_state(payload['torch_rng']); random.setstate(payload['python_rng'])
    # Preserve training RNG when constructing the matched initial control.
    training_rng=torch.get_rng_state()
    initial_policies=make_policies()
    torch.set_rng_state(training_rng)
    for r,a in initial_policies.items(): a.load_state_dict(initial[r])
    before={r:fingerprint(a) for r,a in initial_policies.items()}
    model=load_model(args.device); base_before=fingerprint(model)
    started=time.perf_counter(); session_start=started
    settings=vars(args).copy(); settings['output']=str(root)
    (root/'settings.json').write_text(json.dumps(settings,indent=2))
    if not args.resume:
        save_checkpoint(root/'initial.pt',initial_policies,critic,optimizers,critic_optimizer,initial,0,args.seed,[],0)
    for u in range(completed,args.updates):
        # All three agents remain present throughout the length curriculum.
        bars=args.bars or (4 if u<16 else 8 if u<48 else 16)
        episodes=[]; t=time.perf_counter()
        for e in range(args.episodes):
            seed=100000+args.seed+u*args.episodes+e
            brief=Brief(bars,100,[0,2,5,7,9][(u*args.episodes+e)%5],bool(e%2))
            result=rollout(model,policies,critic,brief,seed)
            episodes.append(result)
            print(f'Update {u+1}: episode {e+1}/{args.episodes}, {len(result["notes"])} notes',flush=True)
        learning=update(policies,critic,optimizers,critic_optimizer,episodes)
        log=dict(update=u+1,bars=bars,episodes=args.episodes,
                 mean_return=sum(sum(s['reward'] for s in ep['steps']) for ep in episodes)/len(episodes),
                 seconds=time.perf_counter()-t,learning=learning)
        logs.append(log); completed=u+1
        total_elapsed=elapsed+time.perf_counter()-started
        save_checkpoint(root/'latest.pt',policies,critic,optimizers,critic_optimizer,initial,completed,args.seed,logs,total_elapsed)
        save_checkpoint(root/f'update_{completed:03}.pt',policies,critic,optimizers,critic_optimizer,initial,completed,args.seed,logs,total_elapsed)
        (root/'training_log.json').write_text(json.dumps(logs,indent=2))
        print(json.dumps(log),flush=True)
        if time.perf_counter()-session_start>=args.max_hours*3600 or total_elapsed>=8*3600: break
    after={r:fingerprint(a) for r,a in policies.items()}
    base_after=fingerprint(model)
    verification=dict(actors_changed={r:before[r]!=after[r] for r in ROLES},
                      frozen_backbone_unchanged=base_before==base_after,
                      backbone_has_no_gradients=all(p.grad is None and not p.requires_grad for p in model.parameters()),
                      completed_updates=completed,training_episodes=sum(x['episodes'] for x in logs),
                      peak_cuda_mib=torch.cuda.max_memory_allocated()/1024**2 if torch.cuda.is_available() else 0)
    assert all(verification['actors_changed'].values()) and verification['frozen_backbone_unchanged']
    # Validation stays separate from training; report it, do not tune on final listening seeds.
    validation=[]
    for seed in [700001,700002]:
        b=Brief(4,100,seed%12,True)
        for name,actors in [('initial',initial_policies),('trained',policies)]:
            ep=rollout(model,actors,critic,b,seed)
            validation.append(dict(seed=seed,method=name,total_return=sum(s['reward'] for s in ep['steps'])))
    (root/'validation.json').write_text(json.dumps(validation,indent=2))
    checks={}
    for name,actors,shared in [('initial',initial_policies,True),('trained',policies,True),('no_partner',policies,False)]:
        ep=rollout(model,actors,critic,Brief(args.demo_bars,100,9,True),900001,shared=shared)
        checks[name]=export_performance(root,name,ep,not args.no_audio)
        print(f'Exported {name}',flush=True)
    verification['exports']=checks
    (root/'verification.json').write_text(json.dumps(verification,indent=2))
    manifest=dict(format_version=1,model='skytnt/midi-model-tv2o-medium',
                  revision='0f8f265d4330f4e46527ac2313200254c5757f5f',
                  independent_actors=True,joint_rl=True,shared_frozen_backbone=True,
                  actor_parameters={r:sum(p.numel() for p in a.parameters()) for r,a in policies.items()},
                  algorithm='MAPPO-style: simultaneous beat proposals, shared state critic, team reward',
                  reward_version='band-heuristic-v1',roles=ROLES,
                  demo_seed=900001,demo_bars=args.demo_bars,
                  quality_improvement_established=False,originality_verified=False,
                  elapsed_seconds=elapsed+time.perf_counter()-started)
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    write_page(root)
    (root/'REPORT.md').write_text('# First joint band training\n\n'
        f'{completed} updates, {verification["training_episodes"]} episodes. Three independent residual policies trained jointly over a frozen backbone.\n\n'
        'All actors choose rest, onset, pitch, velocity and duration; fixed instrument masks define roles. '
        'All see completed past beats; current proposals are private until the shared commit.\n\n'
        'The reward is a heuristic, not a validated quality measure. Validation is descriptive; latest.pt is the latest training state, not a claim of best quality. '
        'The no-partner control removes musical partner observations and is a distribution-shift diagnostic. '
        'This first trial does not establish improved music, long-term structure or originality.\n\n'
        'See verification.json for parameter and MIDI/audio checks, validation.json for fresh paired diagnostics, '
        'and listen.html for the demonstration.\n',encoding='utf-8')
    print(json.dumps(verification,indent=2),flush=True)

def write_page(root):
    # Native media controls and no automatic server dependency.
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
    <title>Three learning musicians</title><style>body{font:18px system-ui;background:#101724;color:#e8edf5;max-width:900px;margin:40px auto;padding:20px}button,select{font:inherit;margin:8px;padding:8px}audio{width:100%}.card{background:#1d293b;padding:20px;margin:20px 0;border-radius:14px}a{color:#9cd6ff}canvas{width:100%;background:#0b111c}</style>
    <h1>Three learning musicians</h1><p>Piano, bass and drums. All notes are generated. This first joint training trial has not established better musical quality.</p>
    <p>Each panel provides an exact mix and its instrument stems. Pause before switching parts.</p>'''
    for method,label in [('trained','After joint training'),('initial','Before training'),('no_partner','Trained policies without partner observations')]:
        page+=f'<section class="card"><h2>{label}</h2><select aria-label="{label} instrument" onchange="switchStem(this)">'
        for stem in ['mix',*ROLES]: page+=f'<option value="{method}_{stem}.mp3">{stem.title()}</option>'
        page+=f'</select><audio controls preload="metadata" src="{method}_mix.mp3"></audio><canvas width="850" height="160" data-events="{method}_events.json"></canvas><p><a href="{method}_mix.mid">Download MIDI</a></p></section>'
    page+='''<script>function switchStem(s){const a=s.parentElement.querySelector('audio');const t=a.currentTime;a.pause();a.src=s.value;a.addEventListener('loadedmetadata',()=>{a.currentTime=Math.min(t,a.duration||0)},{once:true});a.load()}
    document.querySelectorAll('audio').forEach(a=>a.addEventListener('play',()=>document.querySelectorAll('audio').forEach(b=>{if(a!==b)b.pause()})));
    document.querySelectorAll('canvas').forEach(async c=>{try{const notes=await(await fetch(c.dataset.events)).json();const audio=c.parentElement.querySelector('audio');const ctx=c.getContext('2d');const end=Math.max(...notes.map(n=>n[1]+n[2]),1);function draw(){ctx.clearRect(0,0,c.width,c.height);for(const n of notes){ctx.fillStyle=n[3]===0?'#76c9ff':n[3]===1?'#ffc66b':'#ce9dff';ctx.fillRect(n[1]/end*c.width,(95-n[4])/75*c.height,Math.max(2,n[2]/end*c.width),4)}ctx.fillStyle='white';ctx.fillRect(audio.currentTime*800/end*c.width,0,2,c.height);requestAnimationFrame(draw)}draw()}catch{c.remove()}});
    </script></html>'''
    (root/'listen.html').write_text(page,encoding='utf-8')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train'); t.add_argument('--updates',type=int,default=2)
    t.add_argument('--episodes',type=int,default=4); t.add_argument('--bars',type=int)
    t.add_argument('--demo-bars',type=int,default=16); t.add_argument('--max-hours',type=float,default=2)
    t.add_argument('--resume',action='store_true')
    g=sub.add_parser('generate'); g.add_argument('--checkpoint',type=Path,required=True)
    g.add_argument('--bars',type=int,default=16); g.add_argument('--bpm',type=int,default=100)
    g.add_argument('--tonic',type=int,default=9); g.add_argument('--major',action='store_true')
    g.add_argument('--no-partner',action='store_true')
    for parser in [t,g]:
        parser.add_argument('--output',type=Path,default=ROOT/'outputs'/'band_joint_v1')
        parser.add_argument('--seed',type=int,default=20261017)
        parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
        parser.add_argument('--no-audio',action='store_true')
    args=p.parse_args(); torch.set_num_threads(4)
    if args.seed<0: p.error('Seed must be nonnegative')
    if args.command=='train':
        if args.updates<1 or args.episodes<2 or not 0<args.max_hours<=2 or not 1<=args.demo_bars<=16 or (args.bars is not None and not 1<=args.bars<=16):
            p.error('Use positive updates, >=2 episodes, 0..2 hours, and 1..16 bars')
        train(args)
    else: generate(args)

if __name__=='__main__': main()
