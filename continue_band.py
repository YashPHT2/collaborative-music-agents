"""Continue the first band checkpoint with periodic validation and best-state selection."""
import argparse
import copy
import json
import random
import time
from pathlib import Path
import torch
from band import (ROOT,ROLES,Brief,TeamCritic,fingerprint,load_model,rollout,update,export_performance)
from train_band import make_policies,save_checkpoint

VALIDATION=[(710001,0,False),(710002,9,True),(710003,5,False),(710004,2,True)]

def validate(model,actors,critic):
    results=[]
    for seed,tonic,minor in VALIDATION:
        ep=rollout(model,actors,critic,Brief(4,100,tonic,minor),seed)
        bars=[s['components'] for s in ep['steps'] if s['components']]
        results.append(dict(seed=seed,mean_bar_reward=sum(s['reward'] for s in ep['steps'])/4,
            components={k:sum(b[k] for b in bars)/len(bars) for k in bars[0]},
            notes={r:sum(n[3]==s['channel'] for n in ep['notes']) for r,s in ROLES.items()}))
    # Never select a checkpoint that drops a musician on any validation clip.
    valid=all(all(row['notes'].values()) for row in results)
    return dict(mean=sum(r['mean_bar_reward'] for r in results)/len(results),all_agents_present=valid,clips=results)

def choose_best(best,candidate):
    return candidate['all_agents_present'] and (not best['all_agents_present'] or candidate['mean']>best['mean']+1e-6)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'outputs'/'band_joint_v1'/'latest.pt')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs'/'band_joint_v2')
    parser.add_argument('--updates',type=int,default=14)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(); root=args.output
    if (root/'latest.pt').exists() and not args.resume: raise ValueError('Use --resume or a new output folder')
    payload=torch.load(root/'latest.pt' if args.resume else args.source,map_location='cpu',weights_only=False)
    if args.updates<payload['completed']: raise ValueError('Target updates cannot precede the saved update')
    root.mkdir(parents=True,exist_ok=True); torch.set_num_threads(4)
    actors=make_policies(); critic=TeamCritic()
    optimizers={r:torch.optim.Adam(a.parameters(),lr=1e-4) for r,a in actors.items()}
    critic_optimizer=torch.optim.Adam(critic.parameters(),lr=3e-4)
    for r,a in actors.items():
        a.load_state_dict(payload['actors'][r]); optimizers[r].load_state_dict(payload['optimizers'][r])
    critic.load_state_dict(payload['critic']); critic_optimizer.load_state_dict(payload['critic_optimizer'])
    initial=make_policies()
    for r,a in initial.items(): a.load_state_dict(payload['initial'][r])
    torch.set_rng_state(payload['torch_rng']); random.setstate(payload['python_rng'])
    model=load_model(); base_hash=fingerprint(model)
    start_hash={r:fingerprint(a) for r,a in actors.items()}
    seed=payload['seed']; completed=payload['completed']; logs=payload['logs']; started=time.perf_counter()
    starting_update=completed
    if args.resume:
        selection=json.loads((root/'selection.json').read_text())
        best=selection['best']; history=selection['history']
    else:
        base=validate(model,initial,critic); first=validate(model,actors,critic)
        history=[dict(update=0,validation=base),dict(update=completed,validation=first)]
        baseline_wins=not choose_best(base,first)
        best=dict(update=0 if baseline_wins else completed,**(base if baseline_wins else first))
        best_actors=initial if baseline_wins else actors
        save_checkpoint(root/'best.pt',best_actors,critic,optimizers,critic_optimizer,payload['initial'],best['update'],seed,logs,payload['elapsed'])
        (root/'selection.json').write_text(json.dumps(dict(best=best,history=history),indent=2))
        print(f'Initial validation: untrained={base["mean"]:.4f}, first trial={first["mean"]:.4f}',flush=True)
    try:
        for u in range(completed,args.updates):
            # Grow the collaboration horizon after ten updates; all agents train throughout.
            bars=4 if u<10 else 8
            episodes=[]; start=time.perf_counter()
            for e in range(4):
                ep=rollout(model,actors,critic,Brief(bars,100,[0,2,5,7,9][(u*4+e)%5],bool(e%2)),100000+seed+u*4+e)
                episodes.append(ep)
                print(f'Update {u+1}/{args.updates}, episode {e+1}/4, {bars} bars, {len(ep["notes"])} notes',flush=True)
            learning=update(actors,critic,optimizers,critic_optimizer,episodes)
            log=dict(update=u+1,bars=bars,episodes=4,mean_return=sum(sum(s['reward'] for s in ep['steps']) for ep in episodes)/4,
                     mean_bar_reward=sum(sum(s['reward'] for s in ep['steps'])/bars for ep in episodes)/4,
                     seconds=time.perf_counter()-start,learning=learning)
            logs.append(log); completed=u+1
            elapsed=payload['elapsed']+time.perf_counter()-started
            save_checkpoint(root/'latest.pt',actors,critic,optimizers,critic_optimizer,payload['initial'],completed,seed,logs,elapsed)
            save_checkpoint(root/f'update_{completed:03}.pt',actors,critic,optimizers,critic_optimizer,payload['initial'],completed,seed,logs,elapsed)
            (root/'training_log.json').write_text(json.dumps(logs,indent=2))
            print(json.dumps(log),flush=True)
            if (completed-2)%4==0 or completed==args.updates:
                result=validate(model,actors,critic)
                history.append(dict(update=completed,validation=result))
                if choose_best(best,result):
                    best=dict(update=completed,**result)
                    save_checkpoint(root/'best.pt',actors,critic,optimizers,critic_optimizer,payload['initial'],completed,seed,logs,elapsed)
                (root/'selection.json').write_text(json.dumps(dict(best=best,history=history),indent=2))
                print(f'VALIDATION update {completed}: {result["mean"]:.4f}; best update {best["update"]}: {best["mean"]:.4f}',flush=True)
            if time.perf_counter()-started>2*3600: break
    finally:
        verification=dict(completed_updates=completed,training_episodes=sum(x['episodes'] for x in logs),
            all_actors_changed={r:start_hash[r]!=fingerprint(a) for r,a in actors.items()},
            backbone_unchanged=base_hash==fingerprint(model),best_update=best['update'])
        if completed==starting_update and (root/'training_verification.json').exists():
            previous_verification=json.loads((root/'training_verification.json').read_text())
            verification['all_actors_changed']=previous_verification['all_actors_changed']
            verification['export_only_resume']=True
        (root/'training_verification.json').write_text(json.dumps(verification,indent=2))
    assert (completed==starting_update or all(verification['all_actors_changed'].values())) and verification['backbone_unchanged']
    selected=torch.load(root/'best.pt',map_location='cpu',weights_only=False)
    for r,a in actors.items(): a.load_state_dict(selected['actors'][r])
    # Fresh listening seed, never used in checkpoint selection.
    checks={}
    for method,policies in [('selected',actors),('initial',initial)]:
        episode=rollout(model,policies,critic,Brief(16,100,9,True),910001)
        (root/f'{method}_events.json').write_text(json.dumps(episode['notes'],indent=2))
        (root/f'{method}_trace.json').write_text(json.dumps(episode['trace'],indent=2))
        checks[method]=export_performance(root,method,episode)
        print(f'Exported {method} mix and stems',flush=True)
    (root/'verification.json').write_text(json.dumps(checks,indent=2))
    summary=dict(completed_updates=completed,training_episodes=verification['training_episodes'],
        source=str(args.source),selected_update=best['update'],validation=best,
        musical_quality_improvement_established=False,reward_version='band-heuristic-v1',demo_seed=910001,
        model_revision='0f8f265d4330f4e46527ac2313200254c5757f5f',elapsed_seconds=time.perf_counter()-started)
    (root/'manifest.json').write_text(json.dumps(summary,indent=2))
    (root/'REPORT.md').write_text(f'# Continued joint training\n\n{completed} total updates; {summary["training_episodes"]} training episodes. '
        f'Validation selected update {best["update"]}, including the untrained state as a candidate.\n\n'
        'Validation uses four fixed four-bar seeds, separate from training and the new 16-bar listening seed. '
        'All three instruments must be present on every validation clip for selection. '
        'The reward and sampling constraints are unchanged from the first trial. '
        'Higher heuristic reward does not establish better sounding music. '
        'See selection.json for all candidates, training_verification.json for weight checks, and verification.json for audio/MIDI checks.\n',encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': main()
