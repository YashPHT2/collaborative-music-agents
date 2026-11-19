"""Continue the direct note musicians, retaining a validation-selected checkpoint."""
import argparse
import copy
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import note_trio as nt

VALIDATION_SEEDS=[8210001,8210002,8210003,8210004]
FINAL_SEED=9401001
ROOT=nt.ROOT/'outputs'/'note_trio_v2'
SOURCE=nt.ROOT/'outputs'/'note_trio_v1'/'latest.pt'


def metrics(ep):
    bars=ep['brief']['bars']
    role_stats={}
    for role,s in nt.ROLES.items():
        ns=[n for n in ep['notes'] if n[3]==s['channel']]
        pitches=[n[4] for n in ns]
        dominance=max((pitches.count(p) for p in set(pitches)),default=0)/max(1,len(pitches))
        role_stats[role]=dict(notes=len(ns),distinct_pitches=len(set(pitches)),dominant_pitch_share=dominance)
    eligible=all(s['notes']>=bars and s['dominant_pitch_share']<=.85 for s in role_stats.values())
    return dict(mean_bar_reward=float(np.mean([s['reward'] for s in ep['steps'] if s['components']])),
                eligible=eligible,roles=role_stats)


def validate(model,agents,critic):
    cases=[]
    for seed in VALIDATION_SEEDS:
        ep=nt.rollout(model,agents,critic,nt.Brief(4,100,seed%12,True),seed)
        cases.append(dict(seed=seed,**metrics(ep)))
    return dict(score=float(np.mean([c['mean_bar_reward'] for c in cases])),
                eligible=all(c['eligible'] for c in cases),cases=cases)


def should_select(candidate,current):
    return candidate['eligible'] and (not current['eligible'] or candidate['score']>current['score']+1e-6)


def load_into(saved,agents,critic,opts=None,co=None):
    if saved['format_version']!=nt.FORMAT or saved['roles']!=nt.ROLES:
        raise ValueError('Wrong model format/roles')
    for role,a in agents.items():
        a.load_state_dict(saved['actors'][role])
        if opts is not None: opts[role].load_state_dict(saved['optimizers'][role])
    critic.load_state_dict(saved['critic'])
    if co is not None: co.load_state_dict(saved['critic_optimizer'])


def write_page(root,summary):
    nt.write_page(root)
    p=root/'listen.html'; text=p.read_text()
    text=text.replace('Three note-generating musicians','Note musicians — continued training')
    text=text.replace('After joint training',f'Selected checkpoint · update {summary["selected_update"]}')
    text=text.replace('Matched untrained baseline',f'Before continuation · update {summary["starting_update"]}')
    text=text.replace('Trained without partner observations',f'Selected update {summary["selected_update"]} without partner observations')
    text=text.replace('initial training','continued training')
    text=text.replace('This first joint-training result has',
        f'{summary["new_episodes"]} additional joint episodes completed. This performance has')
    text=text.replace('<h1>Note musicians — continued training</h1>',
        '<h1>Note musicians — continued training</h1><p>Checkpoint selection uses a bounded musical-rule score; listening preference has not been established.</p>')
    p.write_text(text,encoding='utf-8')


def include_latest_comparison(model,root=ROOT):
    """Expose actual new training even when validation retains the starting model.

    Avoid presenting identical selected/before audio as a misleading comparison.
    """
    path=root/'verification.json'; summary=json.loads(path.read_text())
    if summary['selected_update']>summary['starting_update']: return
    latest=torch.load(root/'latest.pt',map_location='cpu',weights_only=False)
    agents={r:nt.Musician() for r in nt.ROLES}; critic=nt.TeamCritic()
    load_into(latest,agents,critic)
    ep=nt.rollout(model,agents,critic,nt.Brief(24,100,4,True),FINAL_SEED)
    summary['latest_exports']=nt.export(root,'latest',ep,stems=True)
    summary['latest_stats']=metrics(ep)
    summary['latest_provenance']=dict(notes=len(ep['notes']),direct_policy_decisions=len(ep['decisions']),
        exact_note_provenance=sorted(ep['notes'])==sorted(d['note'] for d in ep['decisions']))
    fallback=summary['selected_update']==summary['starting_update']
    summary['default_playback']='latest' if fallback else 'trained'
    summary['latest_is_experimental']=True
    write_page(root,summary)
    page=root/'listen.html'; text=page.read_text()
    option=f'<option value="latest"{" selected" if fallback else ""}>Latest trained · update {summary["completed_update"]} (experimental)</option>'
    text=text.replace('<select id="method">','<select id="method">'+option)
    text=text.replace("s.disabled=m.value!=='trained'","s.disabled=!['trained','latest'].includes(m.value)")
    if fallback:
        text=text.replace('src="trained_mix.mp3"','src="latest_mix.mp3"')
        text=text.replace(f'<option value="initial">Before continuation · update {summary["starting_update"]}</option>','')
        text=text.replace('<div class="card">','<p>Validation retained the previous checkpoint. The default audio is the latest experimental model so you can hear the actual new training.</p><div class="card">')
    page.write_text(text,encoding='utf-8'); path.write_text(json.dumps(summary,indent=2))
    with (root/'REPORT.md').open('a',encoding='utf-8') as report:
        report.write('\nLatest experimental checkpoint audio is also provided. When the starting model remains selected, duplicate selected/before choices are removed and latest experimental audio is the default.\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--updates',type=int,default=32); p.add_argument('--max-minutes',type=float,default=40)
    p.add_argument('--resume',action='store_true'); args=p.parse_args()
    if args.updates<9 or not 0<args.max_minutes<=480: p.error('Use updates >=9 and <=480 minutes')
    ROOT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    if (ROOT/'latest.pt').exists() and not args.resume: raise ValueError('Continuation exists; use --resume')
    if not (ROOT/'before.pt').exists(): shutil.copy2(SOURCE,ROOT/'before.pt')
    baseline_saved=torch.load(ROOT/'before.pt',map_location='cpu',weights_only=False)
    saved=torch.load(ROOT/'latest.pt' if args.resume else ROOT/'before.pt',map_location='cpu',weights_only=False)
    agents={r:nt.Musician() for r in nt.ROLES}; critic=nt.TeamCritic()
    opts={r:torch.optim.Adam(a.parameters(),lr=1e-4) for r,a in agents.items()}; co=torch.optim.Adam(critic.parameters(),lr=3e-4)
    load_into(saved,agents,critic,opts,co)
    initial,logs,elapsed,settings=saved['initial'],saved['logs'],saved['elapsed'],saved['settings']
    if settings!=baseline_saved['settings']: raise ValueError('Changed continuation settings')
    torch.set_rng_state(saved['torch_rng']); np.random.set_state(saved['numpy_rng']); random.setstate(saved['python_rng'])
    if torch.cuda.is_available() and saved['cuda_rng']: torch.cuda.set_rng_state_all(saved['cuda_rng'])
    model=nt.load_model(); base=nt.fingerprint(model)
    selection_path=ROOT/'selection.json'
    if selection_path.exists(): selection=json.loads(selection_path.read_text())
    else:
        result=validate(model,agents,critic)
        selection=dict(selected_update=len(logs),best=result,checks=[dict(update=len(logs),**result)],no_improvement=0)
        shutil.copy2(ROOT/'before.pt',ROOT/'best.pt')
        selection_path.write_text(json.dumps(selection,indent=2))
        print(f'Starting update {len(logs)} validation: {result["score"]:.4f}; eligible={result["eligible"]}',flush=True)
    started=time.perf_counter(); completed_at_start=len(logs)
    for u in range(len(logs),args.updates):
        if elapsed+time.perf_counter()-started>=8*3600: break
        bars=8 if u<16 else 16
        eps=[]; tick=time.perf_counter()
        for e in range(settings['episodes']):
            brief=nt.Brief(bars,100,[4,9,2,7][(u*settings['episodes']+e)%4],True)
            ep=nt.rollout(model,agents,critic,brief,1100000+u*settings['episodes']+e)
            eps.append(ep)
            print(f'Continuation {u+1}/{args.updates}, {bars} bars, episode {e+1}/{settings["episodes"]}: {len(ep["notes"])} notes',flush=True)
        learning=nt.update(agents,critic,opts,co,eps)
        logs.append(dict(update=u+1,bars=bars,episodes=settings['episodes'],seconds=time.perf_counter()-tick,
                         mean_bar_reward=float(np.mean([s['reward'] for ep in eps for s in ep['steps'] if s['components']])),learning=learning))
        total_elapsed=elapsed+time.perf_counter()-started
        nt.checkpoint(ROOT/'latest.pt',agents,critic,opts,co,initial,logs,settings,total_elapsed)
        (ROOT/'training_log.json').write_text(json.dumps(logs,indent=2))
        due=(u+1)%8==0 or u+1==args.updates or time.perf_counter()-started>=args.max_minutes*60
        if due:
            result=validate(model,agents,critic)
            selection['checks'].append(dict(update=u+1,**result))
            if should_select(result,selection['best']):
                selection.update(selected_update=u+1,best=result,no_improvement=0)
                shutil.copy2(ROOT/'latest.pt',ROOT/'best.pt')
            else: selection['no_improvement']+=1
            shutil.copy2(ROOT/'latest.pt',ROOT/f'update_{u+1:03}.pt')
            selection_path.write_text(json.dumps(selection,indent=2))
            print(f'Validation update {u+1}: {result["score"]:.4f}, eligible={result["eligible"]}; selected {selection["selected_update"]}',flush=True)
            if selection['no_improvement']>=2: print('Stopping after two validation checks without improvement.',flush=True); break
        if time.perf_counter()-started>=args.max_minutes*60: break
    best=torch.load(ROOT/'best.pt',map_location='cpu',weights_only=False)
    load_into(best,agents,critic)
    before={r:nt.Musician() for r in nt.ROLES}; before_critic=nt.TeamCritic(); load_into(baseline_saved,before,before_critic)
    exports={}; stats={}; provenance={}
    for name,policies,c,shared in [('trained',agents,critic,True),('initial',before,before_critic,True),('no_partner',agents,critic,False)]:
        ep=nt.rollout(model,policies,c,nt.Brief(24,100,4,True),FINAL_SEED,shared=shared)
        exports[name]=nt.export(ROOT,name,ep,stems=name=='trained'); stats[name]=metrics(ep)
        provenance[name]=dict(notes=len(ep['notes']),direct_policy_decisions=len(ep['decisions']),
                             exact_note_provenance=sorted(ep['notes'])==sorted(d['note'] for d in ep['decisions']))
        print(f'Exported {name}',flush=True)
    latest=torch.load(ROOT/'latest.pt',map_location='cpu',weights_only=False)
    summary=dict(starting_update=len(baseline_saved['logs']),completed_update=len(logs),selected_update=selection['selected_update'],
                 total_episodes=sum(x['episodes'] for x in logs),new_episodes=sum(x['episodes'] for x in logs[len(baseline_saved['logs']):]),
                 frozen_backbone_unchanged=nt.fingerprint(model)==base,
                 all_latest_actors_changed={r:any(not torch.equal(v,baseline_saved['actors'][r][k]) for k,v in a.items()) for r,a in latest['actors'].items()},
                 selected_validation=selection['best'],final_seed=FINAL_SEED,exports=exports,stats=stats,provenance=provenance,
                 no_scripted_notes=True,learned_human_preference=False,musical_quality_improvement_established=False)
    assert summary['frozen_backbone_unchanged'] and all(summary['all_latest_actors_changed'].values())
    (ROOT/'verification.json').write_text(json.dumps(summary,indent=2)); write_page(ROOT,summary)
    (ROOT/'REPORT.md').write_text('# Continued note-agent training\n\n'
        f'Continued from update {summary["starting_update"]} to {summary["completed_update"]}; selected update {summary["selected_update"]}. '
        f'{summary["new_episodes"]} new joint episodes. The same note-level sampler, musical-rule reward and PPO settings were retained. '
        'Training horizons grew from 8 to 16 bars. Checkpoint selection used four separate validation seeds and required all roles to participate without single-pitch collapse. '
        'The starting model remained eligible. The fresh 60-second listening seed did not select the checkpoint. '
        'Every note was emitted by a musician policy. No melody, chord progression, phrase bank, copy controller or crescendo was supplied. '
        'Validation scores are heuristic diagnostics; improved listening preference is not established.\n',encoding='utf-8')
    include_latest_comparison(model)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['stats','exports','selected_validation']},indent=2),flush=True)


if __name__=='__main__': main()
