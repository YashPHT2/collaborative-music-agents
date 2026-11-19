"""Twelve fixed seeds, held-out response probes and three unselected listening clips."""
import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
import ensemble_repair as er
import offline_ensemble as oe
import note_trio as nt
from partner_response import ensemble_score
from train_ensemble_repair import OUT, corpus, write_json

SEEDS=list(range(9619101,9619113))


def summarize_probes(cases):
    result={}
    for role in er.ROLES:
        rows=[r for r in cases if r['role']==role]
        differences={k:[r['scores']['correct']-r['scores'][k] for r in rows] for k in ('missing','mismatched')}
        confidence={};rng=random.Random(972001)
        for k,values in differences.items():
            boots=sorted(float(np.mean(rng.choices(values,k=len(values)))) for _ in range(1000)) if values else [0]*1000
            confidence[k]=[boots[25],boots[974]]
        change=float(np.mean([r['scores']['correct']-r['baseline_correct'] for r in rows])) if rows else None
        means={k:float(np.mean(v)) if v else None for k,v in differences.items()}
        result[role]=dict(compositions=len({r['group'] for r in rows}),mean_advantage=means,bootstrap_95_intervals=confidence,
                          change_from_initial=change,passed=len(rows)>=10 and change>=0 and all(v>0 for v in means.values()))
    return result


def page(root,items):
    identity=hashlib.sha256(json.dumps(items,sort_keys=True).encode()).hexdigest()
    write_json(root/'panel_identity.json',dict(sha256=identity))
    write_json(root/'panel_key.json',items)
    template='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trio repair listening</title><style>body{font:18px system-ui;background:#101724;color:#e8edf5;max-width:850px;margin:40px auto;padding:20px}p{line-height:1.6}audio{width:100%;margin:15px 0}button,select,input{font:inherit;padding:9px;margin:6px;max-width:100%}label{display:block}.card{background:#1c293d;padding:20px;border-radius:12px}a{color:#9bd2ff}</style>
<h1>Three short trio performances</h1><p>Listen to each piece on its own. Each uses the same seed, length and sound library. The agents choose every note. The build option changes density, register and dynamics targets; its emotional effect is for you to judge.</p>
<p id="selection"></p><div class="card"><label>Piece <select id="clip"></select></label><label>Listen to <select id="stem"><option value="mix">All three together</option><option value="organ">Organ</option><option value="strings">Strings</option><option value="cello">Cello</option></select></label><audio id="audio" controls preload="metadata"></audio><button id="restart">Play from beginning</button><p id="notes"></p><p>Solo levels are adjusted for listening. Rate the full trio.</p></div>
<label>Your name or nickname <input id="voter" value="Anonymous"></label><div id="scores"></div><button id="save">Save this rating</button><button id="download">Download all ratings</button><p id="status"></p><p><a href="evaluation.json">Evaluation report</a> · <a href="REPORT.md">What changed and what remains unproven</a></p>
<script>const clips=__PANEL__,panelHash=__HASH__,storage='trio-repair-'+panelHash,ratings=JSON.parse(localStorage.getItem(storage)||'{}'),clip=document.getElementById('clip'),stem=document.getElementById('stem'),audio=document.getElementById('audio'),status=document.getElementById('status'),dimensions=['enjoyment','development','ensemble_fit','warmth'];
clips.forEach(c=>clip.add(new Option(c.label,c.id)));dimensions.forEach(k=>document.getElementById('scores').insertAdjacentHTML('beforeend',`<label>${{enjoyment:'Enjoyment',development:'Musical development',ensemble_fit:'How well the parts fit',warmth:'Warmth / hopefulness'}[k]} <select id="${k}"><option value="">Choose 1–5</option>${[1,2,3,4,5].map(n=>`<option value="${n}">${n}</option>`).join('')}</select></label>`));
function load(reset=true){const c=clips.find(c=>c.id===clip.value);audio.pause();audio.src=c.id+'_'+stem.value+'.mp3';audio.load();document.getElementById('notes').textContent=c.summary;if(reset)dimensions.forEach(k=>document.getElementById(k).value=ratings[c.id]?.[k]||'');}clip.onchange=()=>{stem.value='mix';load()};stem.onchange=()=>load(false);load();
document.getElementById('restart').onclick=()=>{audio.currentTime=0;audio.play().catch(e=>status.textContent='Press the audio play button to start.')};
document.getElementById('save').onclick=()=>{const values=Object.fromEntries(dimensions.map(k=>[k,+document.getElementById(k).value]));if(stem.value!=='mix'){status.textContent='Switch to All three together before rating.';return;}if(Object.values(values).some(v=>v<1||v>5)){status.textContent='Choose a score for each question.';return;}ratings[clip.value]=values;localStorage.setItem(storage,JSON.stringify(ratings));status.textContent=Object.keys(ratings).length+'/3 pieces rated.'};
document.getElementById('download').onclick=()=>{if(Object.keys(ratings).length!==clips.length){status.textContent='Please rate all three pieces first.';return;}const a=document.createElement('a'),url=URL.createObjectURL(new Blob([JSON.stringify({version:3,panel_sha256:panelHash,voter:document.getElementById('voter').value,ratings},null,2)],{type:'application/json'}));a.href=url;a.download='trio-repair-votes.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
document.getElementById('selection').textContent=__SELECTION__;</script></html>'''
    log=json.loads((root/'training_log.json').read_text())
    selected='Selected training updates: '+', '.join(f'{k} {v["step"]}' for k,v in log['selections'].items())+'. Listener improvement has not yet been established.'
    (root/'listen.html').write_text(template.replace('__PANEL__',json.dumps(items)).replace('__HASH__',json.dumps(identity)).replace('__SELECTION__',json.dumps(selected)),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args();root=args.output
    training=json.loads((root/'training_log.json').read_text())
    saved=torch.load(root/'latest.pt',map_location='cpu',weights_only=False)
    elapsed_file=root/'evaluation_time.json'
    prior=json.loads(elapsed_file.read_text())['seconds'] if elapsed_file.exists() else 0
    started=time.monotonic();limit=saved['config']['hours']*3600-training['elapsed']
    def budget():
        elapsed=prior+time.monotonic()-started;write_json(elapsed_file,dict(seconds=elapsed))
        if elapsed>=limit:raise TimeoutError('Shared three-hour budget reached; partial results retained')
    torch.set_num_threads(4)
    windows,groups,audit,_=corpus()
    audit_file=root/'data_audit.json'
    data_audit=json.loads(audit_file.read_text())
    data_audit['beat_labels']=audit
    data_audit['missing_definition']='No matching source program. excluded_low_support is a known instrument with fewer than eight projected window notes, excluded from imitation.'
    write_json(audit_file,data_audit)
    model,adapters,agents,planner,conditions=er.load(root/'latest.pt')
    mode=saved['composition_mode'];quantiles=saved['profile_quantiles']
    def restore(checkpoint):
        adapters.load(checkpoint['adapters']);planner.load_state_dict(checkpoint['planner']);conditions.load_state_dict(checkpoint['conditions'])
    metrics=[];items=[]
    old=torch.load(nt.ROOT/'outputs'/'offline_ensemble_v1'/'revised.pt',map_location='cpu',weights_only=False)
    oldplanner=oe.Planner().to(model.device);oldplanner.load_state_dict(old['planner']);oldplanner.eval()
    oldconditions=oe.Conditions().to(model.device);oldconditions.load_state_dict(old['conditions']);oldconditions.eval()
    for variant in ('baseline','repaired','warm_build'):
        if variant=='baseline':adapters.load(old['adapters'])
        else:restore(saved)
        for seed in SEEDS:
            budget();artifact=root/f'{variant}_{seed}_episode.json'
            if artifact.exists():ep=json.loads(artifact.read_text())
            else:
                ep=oe.compose(model,adapters,agents,oldplanner,oldconditions,8,seed,'revised') if variant=='baseline' else er.compose(model,adapters,agents,planner,conditions,8,seed,mode,
                    profile='warm_build' if variant=='warm_build' else None,quantiles=quantiles)
                write_json(artifact,ep)
            # Commit and MIDI writers assert range, cap, overlaps and exact note identity.
            ledger=[]
            with oe.limits():
                for beat in range(32):
                    ledger=nt.commit(ledger,{r:[n for n in ep['notes'] if n[3]==i and n[1]//480==beat] for i,r in enumerate(er.ROLES)},beat,nt.Brief(8))
            nt.midi_file(root/f'{variant}_{seed}_verified.mid',ep['notes'],bpm=100)
            row=dict(variant=variant,seed=seed,participation=er.participation(ep['notes']),note_constraint_violations=0,midi_matches_actions=True)
            metrics.append(row);write_json(root/'seed_metrics.json',metrics)
            if seed==SEEDS[0]:
                checks=nt.export(root,variant,ep,stems=True,allow_silent_stems=True)
                trace=[]
                for segment in ep['trace']:
                    for record in segment.get('records',[]):trace.append(record)
                diagnosis={r:dict(committed_notes=row['participation'][r]['notes'],
                                  proposed_notes=sum(t['proposed'] for t in trace if t['role']==r) if trace else None,
                                  forced_rest_beats=sum(not t['legal_onset'] for t in trace if t['role']==r) if trace else None,
                                  voluntary_stop_beats=sum(t['stopped'] for t in trace if t['role']==r) if trace else None,
                                  audio=checks[r]) for r in er.ROLES}
                write_json(root/f'{variant}_participation_trace.json',diagnosis)
                items.append(dict(id=variant,label={'baseline':'1. Current baseline','repaired':'2. Repaired ensemble','warm_build':'3. Repaired + gradual build'}[variant],
                    seed=seed,summary=' · '.join(f'{r}: {row["participation"][r]["notes"]} notes' for r in er.ROLES),
                    audio_sha256=hashlib.sha256((root/f'{variant}_mix.mp3').read_bytes()).hexdigest(),audio_checks=checks))
            print('Evaluated',variant,seed,{r:row['participation'][r]['notes'] for r in er.ROLES},flush=True)
    # Real-model invariants, including regeneration under reversed execution order.
    restore(saved)
    normal=er.compose(model,adapters,agents,planner,conditions,2,972099,mode)
    reverse=er.compose(model,adapters,agents,planner,conditions,2,972099,mode,order=list(reversed(er.ROLES)))
    if normal['notes']!=reverse['notes']:raise ValueError('Agent order changed notes')
    if adapters.base_hash()!=saved['base_sha256']:raise ValueError('Frozen backbone changed')
    write_json(root/'integration_checks.json',dict(agent_order_independent=True,backbone_unchanged=True,midi_roundtrip=True,all_listening_notes_are_actions=True))
    # Publish the clips immediately; probes update their linked evaluation report.
    write_json(root/'evaluation.json',dict(quality='pending listener ratings',partner_response='running',seed_metrics='seed_metrics.json'))
    (root/'REPORT.md').write_text('# Trio repair\n\nTraining selections: '+json.dumps(training['selections'])+'\n\nAll three clips use seed 9619101, eight bars, the same renderer and mastering. No seed was selected for sounding good. Warm-build controls density/register/velocity targets; emotional quality remains unverified.\n\nThe baseline is the previous v2 revised-mode performance. Partner probes compare against its v3-upgraded initial weights under the same revised causal schedule, not against a different prefix or different partner notes.\n\nSee seed_metrics.json for all 12 seeds per condition and data_audit.json for real validation participation. No quality or cooperation claim follows from a lower imitation loss alone.\n',encoding='utf-8')
    page(root,items)
    response_path=root/'response_probes.json';response=json.loads(response_path.read_text()) if response_path.exists() else []
    initial=torch.load(root/'initial.pt',map_location='cpu',weights_only=False)
    for role in er.ROLES:
        keys=sorted(g for g,ws in groups['test'][role].items() if any(len(w['available_roles'])>=2 for w in ws))
        random.Random(972001+er.ROLES.index(role)).shuffle(keys)
        for index,group in enumerate(keys[:10]):
            if any(r['role']==role and r['group']==group for r in response):continue
            w=next(w for w in groups['test'][role][group] if len(w['available_roles'])>=2)
            seed=973000+index;budget();restore(initial)
            ledger,_=er.fixed_response(model,adapters,agents,planner,conditions,w,role,seed)
            baseline=ensemble_score(ledger,7680,w['available_roles'])[0]
            restore(saved);case=dict(role=role,group=group,song=w['song'],split='test',seed=seed,baseline_correct=baseline,scores={})
            for context in ('correct','missing','mismatched'):
                budget();ledger,notes=er.fixed_response(model,adapters,agents,planner,conditions,w,role,seed,context)
                case['scores'][context]=ensemble_score(ledger,7680,w['available_roles'])[0]
            response.append(case);write_json(response_path,response)
            print('Partner probe',role,index+1,case['scores'],flush=True)
    summary=summarize_probes(response)
    write_json(root/'evaluation.json',dict(quality='pending listener ratings',partner_response=summary,
                cooperation_gate=all(v['passed'] for v in summary.values()),seed_metrics='seed_metrics.json',
                baseline_probe='v2 source weights upgraded to v3 causal scheduling',selections=training['selections']))
    write_json(root/'campaign_status.json',dict(stage='awaiting_listener_ratings',listen='listen.html',quality_improvement_established=False))
    budget();print('Listening panel ready:',root/'listen.html',flush=True)


if __name__=='__main__':main()
