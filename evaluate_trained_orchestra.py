"""Matched initial/trained calm orchestral previews after the supervised run."""
import os
os.environ.setdefault('USE_TF','0')
os.environ.setdefault('USE_FLAX','0')
from concurrent.futures import ThreadPoolExecutor
import copy
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time
import argparse

import numpy as np
import soundfile as sf
import torch

import orchestra_learning as l
import generate_full_orchestra as renderer
import generate_focus_duet as felt
import note_trio as nt
import offline_ensemble as oe
import ensemble_repair as er
from audit_projected_roles import loudness
from train_ensemble_repair import write_json
from train_orchestra_learning import load_data,fixed_windows,eligible

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'outputs/orchestra_learning_v1'
OUT=ROOT/'outputs/orchestra_focus_comparison_v1'
SEED=9919101
BARS=8
BPM=70
SECONDS=32*round(60_000_000/BPM)/1_000_000+3


def intervention(notes,role,context):
    out=copy.deepcopy(notes);own=l.ROLES[role]['id']
    if context=='missing':return [n for n in out if n[3]==own]
    if context=='mismatched':
        for n in out:
            if n[3]!=own:n[4]=min(127,n[4]+6)
    return out


@torch.no_grad()
def response(model,adapters,agents,planner,conditions,w,role,seed,context):
    channel=l.ROLES[role]['id'];ledger=l.visible(w['notes'],8)
    fixed=[n for n in w['notes'] if n[3]!=channel and 3840<=n[1]<7680]
    goal=torch.tensor(w['style'],device=model.device);observed=intervention(ledger,role,context)
    plans=l.plan_sequence(planner,observed,6,w['available_roles'],goal)
    features=l.draft_features(intervention(fixed,role,context),role,8).to(model.device);made=[]
    adapters.activate(l.ROLES[role]['family'])
    for beat in range(8,16):
        delta=conditions(plans[beat//4],features,role,goal);past=intervention(l.visible(ledger,beat),role,context)
        sampler=l.Sampler(oe.ConditionedModel(model,delta),role,seed+beat*1009,nt.Brief(6,70,0,False),beat,past)
        notes,_=sampler.propose(agents[role],torch.zeros(32))
        for n in notes:n[3]=channel
        actions={r:[n for n in fixed if n[3]==s['id'] and n[1]//480==beat] for r,s in l.ROLES.items()}
        actions[role]=notes;ledger=l.commit(ledger,actions,beat,6);made.extend(notes)
    assert sorted(n for n in ledger if n[3]!=channel and n[1]>=3840)==sorted(fixed)
    consonant=overlap=0
    for a in [n for n in ledger if n[3]==channel]:
        for b in [n for n in ledger if n[3]!=channel]:
            ticks=max(0,min(7680,a[1]+a[2],b[1]+b[2])-max(3840,a[1],b[1]));overlap+=ticks
            consonant+=ticks*(abs(a[4]-b[4])%12 in [0,3,4,5,7,8,9])
    sounding=sum(any(n[3]==channel and n[1]<=t<n[1]+n[2] for n in ledger) for t in range(3840,7680,30))/16
    return dict(generated=made,notes=len(made),sounding_beats=sounding,consonant_overlap=None if not overlap else consonant/overlap,
                actual_partners_unchanged=True)


def participation(notes,beats=32):
    result={}
    for role,s in l.ROLES.items():
        own=[n for n in notes if n[3]==s['id']];active=np.zeros(beats*16,dtype=bool)
        for n in own:active[n[1]//30:min(len(active),(n[1]+n[2])//30)]=True
        longest=run=0
        for sounding in active:run=0 if sounding else run+1;longest=max(longest,run)
        result[role]=dict(notes=len(own),sounding_beats=float(active.sum()/16),longest_silence_beats=longest/16)
    return result


def render(variant,episode,mix_profile='focus'):
    directory=OUT/variant;directory.mkdir(exist_ok=True)
    renderer.OUT=directory;renderer.SECONDS=SECONDS;renderer.TEMPO=round(60_000_000/BPM)
    renderer.export_midis(episode)
    piano_ids={s['id'] for s in l.ROLES.values() if s['family']=='pianos'}
    piano=[n for n in episode['notes'] if n[3] in piano_ids]
    felt.OUT=directory;felt.SECONDS=SECONDS;felt.BPM=BPM
    if piano:piano_audio=felt.render_piano(piano,felt.sample_assets(piano))
    else:piano_audio=np.zeros((round(SECONDS*felt.RATE),2))
    # Felt sources are 48 kHz; render the remaining orchestra at the same rate.
    renderer.RATE=felt.RATE
    families={f:np.zeros_like(piano_audio) for f in l.FAMILIES};families['pianos']=piano_audio
    other=[r for r,s in l.ROLES.items() if s['family']!='pianos']
    with ThreadPoolExecutor(max_workers=3) as pool:
        for role,audio in pool.map(renderer.render_role,other):families[l.ROLES[role]['family']]+=audio
    piano_rms=float(np.sqrt(np.mean(piano_audio**2)))
    fallback=max((float(np.sqrt(np.mean(v*v))) for v in families.values()),default=0)*.25
    reference=max(piano_rms,fallback,1e-9)
    levels=dict(organ=-16,strings=-12,woodwinds=-18,brass=-22,synth=-20,choir=-20)
    if mix_profile=='organ_lead':levels=dict(organ=4,strings=-6,woodwinds=-16,brass=-18,synth=-18,choir=-18)
    elif mix_profile!='focus':raise ValueError('Unknown mix profile: '+mix_profile)
    gains={}
    for family,db in levels.items():
        ceiling=16. if mix_profile=='organ_lead' and family=='organ' else 1.
        rms=float(np.sqrt(np.mean(families[family]**2)));gain=min(ceiling,reference*10**(db/20)/max(rms,1e-12))
        families[family]*=gain;gains[family]=gain
    dry=sum(families.values());wet=renderer.convolution_room(dry);mix=dry+.06*wet
    fade=np.ones(len(mix));fade[-felt.RATE:]=np.linspace(1,0,felt.RATE);mix*=fade[:,None]
    path=directory/'mix_raw.wav';sf.write(path,mix,felt.RATE,subtype='FLOAT')
    write_json(directory/'render_recipe.json',dict(family_gain=gains,piano_sources='Authentic Meta felt-damped pp/mp OGG samples',
        piano_rms=piano_rms,sampled_orchestra=True,quiet_background_balance=levels,mix_profile=mix_profile,convolution_wet=.06,source_note_count=len(episode['notes'])))
    return path,families,fade


def page():
    (OUT/'listen.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,"><title>Trained orchestra â€” calmer focus comparison</title><style>body{font:18px system-ui;line-height:1.6;background:#111d23;color:#ecf2f4;max-width:950px;margin:35px auto;padding:20px}section{background:#203239;border-radius:14px;padding:24px;margin:24px 0}audio{width:100%;margin:15px 0}button,select{font:inherit;padding:10px}label{display:block;margin:15px 0}a{color:#b2e3d1}</style><h1>Did ensemble training help?</h1><p>Two new 30-second performances using the same seed, quiet C-major controls, instruments and matched loudness. Felt piano leads; organ, strings, winds, brass, synth and sampled choir provide quieter support.</p><section><h2>Before this training</h2><audio controls preload="metadata" src="initial.mp3"></audio><button>Play from beginning</button></section><section><h2>After ensemble training</h2><audio controls preload="metadata" src="trained.mp3"></audio><button>Play from beginning</button></section><p id="selection"></p><label>Which would you actually study or work with? <select id="choice"><option value="">Choose</option><option value="initial">Before training</option><option value="trained">After training</option><option value="neither">Neither</option><option value="same">Both equally</option></select></label><label>Does the trained music feel scary? <select id="mood"><option value="">Choose</option><option>Yes</option><option>A little</option><option>No</option></select></label><button id="save">Download feedback</button><p id="status" role="status"></p><p><a href="evaluation.json">Training selections and diagnostics</a> Â· <a href="README.md">What was trained</a></p><p>These are sampled instruments. Musical enjoyment, positive mood and concentration benefits require your feedback; this run does not establish them.</p><script>for(const card of document.querySelectorAll('section')){const a=card.querySelector('audio');card.querySelector('button').onclick=()=>{a.currentTime=0;a.play().catch(()=>{})};a.addEventListener('play',()=>{for(const other of document.querySelectorAll('audio'))if(other!==a)other.pause()})}fetch('evaluation.json').then(r=>r.json()).then(e=>document.getElementById('selection').textContent=e.selected_checkpoint_passed?'The trained clip uses a checkpoint that passed the held-out prediction checks.':'The trained clip is an experimental candidate; the strict checkpoint selection did not establish an improvement.');document.getElementById('save').onclick=()=>{const choice=document.getElementById('choice').value,mood=document.getElementById('mood').value;if(!choice||!mood){document.getElementById('status').textContent='Please answer both questions.';return}const url=URL.createObjectURL(new Blob([JSON.stringify({schema:'orchestra-focus-v1',seed:9919101,choice,trained_scary:mood},null,2)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download='orchestra-focus-feedback.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};</script></html>'''.replace('seed:9919101','seed:'+str(SEED)),encoding='utf-8')


def comparison_inputs(run,checkpoint=None,initial_checkpoint=None):
    """Accept explicit in-progress snapshots without marking training complete."""
    log=json.loads((run/'training_log.json').read_text())
    initial_path=initial_checkpoint or run/'initial.pt'
    initial=torch.load(initial_path,map_location='cpu',weights_only=False)
    metadata=dict(training_complete=bool(log.get('complete')),explicit_snapshot=checkpoint is not None)
    if checkpoint is None:
        if not log.get('complete'):raise ValueError('Training has not finished; use --checkpoint to preview a saved snapshot')
        passed=any(v['passed'] for v in log['selections'].values())
        candidates=[run/'partner_candidate.pt',run/'phrases_candidate.pt',run/'initial.pt']
        trained_path=run/'selected.pt' if passed else next(path for path in candidates if path.exists())
    else:
        trained_path=checkpoint
        stage='partner' if trained_path.stem.startswith('partner') else 'phrases'
        selected=log.get('selected_step',0) if log.get('stage')==stage else log.get('selections',{}).get(stage,{}).get('step',0)
        evidence=next((v for v in log.get('validations',[]) if v['stage']==stage and v['step']==selected),None)
        passed=bool(evidence and evidence['passed'])
        if evidence and log.get('stage')==stage:
            passed=passed and eligible(evidence['scores'],log['initial'])
        metadata.update(snapshot_stage=stage,log_selected_step=selected,log_step=log.get('step'),
                        validation_evidence='Downloaded log reports the selected step; weights have no embedded step counter.')
    trained=torch.load(trained_path,map_location='cpu',weights_only=False)
    for field in ['format_version','base_sha256','roles','families']:
        if initial[field]!=trained[field]:raise ValueError('Checkpoint compatibility mismatch: '+field)
    expected=trained.get('h100_config',{}).get('init_sha256')
    if expected and hashlib.sha256(initial_path.read_bytes()).hexdigest()!=expected:
        raise ValueError('Comparison baseline is not the initialization used by this training run')
    return log,initial,trained,initial_path,trained_path,passed,metadata


def main():
    global RUN,OUT,SEED
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=RUN);parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--checkpoint',type=Path,help='Explicit saved snapshot; does not require completed training')
    parser.add_argument('--initial-checkpoint',type=Path,help='Exact initialization used for matched before/after comparison')
    parser.add_argument('--skip-probes',action='store_true');parser.add_argument('--seed',type=int,default=SEED)
    args=parser.parse_args();RUN=args.run.resolve();OUT=args.output.resolve();SEED=args.seed
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    log,initial,trained,initial_path,trained_path,passed,snapshot=comparison_inputs(RUN,args.checkpoint,args.initial_checkpoint)
    protocol=dict(seed=SEED,bars=BARS,bpm=BPM,source_checkpoints={k:hashlib.sha256(path.read_bytes()).hexdigest() for k,path in [('initial',initial_path),('trained',trained_path)]},
                  trained_checkpoint=str(trained_path),selected_checkpoint_passed=passed,calm_profile=trained['calm_profile'],**snapshot)
    write_json(OUT/'protocol.json',protocol)
    model=nt.load_model();adapters=l.Adapters(model);planner=l.Planner().to(model.device);conditions=l.Conditions().to(model.device)
    agents={r:oe.Musician().eval() for r in l.ROLES};base_hash=adapters.base_hash()
    if base_hash!=trained['base_sha256']:raise ValueError('Local backbone does not match the checkpoint')
    def restore(saved):
        adapters.load(saved['adapters']);planner.load_state_dict(saved['planner']);conditions.load_state_dict(saved['conditions']);planner.eval();conditions.eval()
    episodes={};renders={};family_parts={}
    for variant,saved in [('initial',initial),('trained',trained)]:
        print('Generating comparison:',variant,flush=True)
        restore(saved);ep=l.compose(model,adapters,agents,planner,conditions,BARS,SEED,saved['composition_mode'],trained['calm_profile'])
        episodes[variant]=ep;write_json(OUT/(variant+'_episode.json'),ep)
        path,families,fade=render(variant,ep);renders[variant]=path;family_parts[variant]=(families,fade)
        print('Rendered comparison:',variant,len(ep['notes']),'agent notes',flush=True)
    levels={k:loudness(p) for k,p in renders.items()};target=min([-22]+[-2-(peak-integrated) for integrated,peak in levels.values()])
    checks={}
    for variant,path in renders.items():
        gain=target-levels[variant][0];wav=OUT/(variant+'.wav');mp3=OUT/(variant+'.mp3')
        subprocess.run(['ffmpeg','-y','-v','error','-i',str(path),'-af',f'volume={gain:.9f}dB','-c:a','pcm_s24le',str(wav)],check=True)
        subprocess.run(['ffmpeg','-y','-v','error','-i',str(wav),'-q:a','2',str(mp3)],check=True)
        decoded=OUT/(variant+'_decoded.wav');subprocess.run(['ffmpeg','-y','-v','error','-i',str(mp3),str(decoded)],check=True)
        audio,sr=sf.read(decoded);decoded.unlink();assert np.isfinite(audio).all() and np.max(np.abs(audio))<1
        checks[variant]=dict(seconds=len(audio)/sr,decoded_peak=float(np.max(np.abs(audio))),measured_lufs=loudness(wav)[0],note_participation=participation(episodes[variant]['notes']))
    restore(trained);normal=l.compose(model,adapters,agents,planner,conditions,2,SEED+99,trained['composition_mode'],trained['calm_profile'])
    reverse=l.compose(model,adapters,agents,planner,conditions,2,SEED+99,trained['composition_mode'],trained['calm_profile'],order=list(reversed(l.ROLES)))
    assert normal['notes']==reverse['notes'] and adapters.base_hash()==base_hash
    evaluation=dict(status='partner_probes_running',selected_checkpoint_passed=passed,selections=log['selections'],audio=checks,
                    cooperation_established=False,musical_preference='unknown',all_notes_match_actions=True,agent_order_independent=True,**snapshot)
    write_json(OUT/'evaluation.json',evaluation)
    (OUT/'README.md').write_text('''# Trained focus orchestra comparison

The supervised run trains seven rank-8 family LoRA adapters, nineteen role
conditions, a causal phrase planner and an announced-partner encoder over the
unchanged 233M pretrained Transformer. Real aligned partial ensembles provide
four-bar targets with two preceding bars of context. Missing roles have no note
loss. Related piano and violin slots share family adapters, with one real source
stream assigned to each present role slot; one part is never cloned inside a
phrase. This does not provide nineteen-part full-orchestra ground truth.

Stage one learns phrase note prediction; stage two uses real partner drafts and
25% frozen generated drafts on matched-prefix, role-balanced training anchors.
Held-out teacher validation uses oracle announced partner parts and reports
event/stopping and note attributes separately. It is not a free-running taste test.

The previews use the same predetermined seed, 70 BPM, role constraints and calm
profile. Density and velocity control values come from training-only quantiles;
the C-major pitch mask is explicit inference control. No human mood labels were
invented. Real felt piano samples and quiet sampled background sections change
the sound from the earlier General MIDI-only prototype, so comparisons here use
the new rendering for both initial and trained policies. Neither preview copies
music from a reference recording. Lower validation loss alone proves neither
positive mood nor better concentration. No RL, taste reward or DPO is trained.

One fixed untouched test composition per role supports exploratory correct,
missing and tritone-mismatched partner probes with unchanged actual partners.
Nineteen single-composition cases are not strong statistical cooperation evidence.
''',encoding='utf-8');page()
    if snapshot['explicit_snapshot']:
        p=OUT/'listen.html';html=p.read_text(encoding='utf-8')
        html=html.replace('Before this training','Laptop checkpoint: before A100 training').replace('After ensemble training','Downloaded A100 phrase checkpoint')
        banner='<p><strong>In-progress snapshot preview.</strong> The downloaded log reports selected phrase step '+str(snapshot.get('log_selected_step'))+' and training step '+str(snapshot.get('log_step'))+'. Remote training was not stopped. Partner-response training has not been evaluated in this preview.</p>'
        html=html.replace('<h1>Did ensemble training help?</h1>','<h1>Laptop vs A100 phrase training</h1>'+banner)
        p.write_text(html,encoding='utf-8')
    print('New trained listening comparison is ready',flush=True)
    if args.skip_probes:
        evaluation.update(status='listening_ready_probes_skipped',partner_cases=0)
        write_json(OUT/'evaluation.json',evaluation);return
    windows,groups,data=load_data();cases=[]
    for role,s in l.ROLES.items():
        pool=fixed_windows(groups['test'],role,1)
        if not pool:cases.append(dict(role=role,unavailable=True));continue
        w=pool[0];case=dict(role=role,group=w['group'],window_id=w['id'],contexts={});seed=SEED+1000+s['id']
        restore(initial);case['initial_correct']=response(model,adapters,agents,planner,conditions,w,role,seed,'correct')
        restore(trained)
        for context in ['correct','missing','mismatched']:case['contexts'][context]=response(model,adapters,agents,planner,conditions,w,role,seed,context)
        cases.append(case);write_json(OUT/'partner_cases.json',cases);print('Held-out response probe:',role,flush=True)
    evaluation.update(status='complete',partner_cases=len(cases),cooperation_established=False)
    write_json(OUT/'evaluation.json',evaluation);print('EVALUATION COMPLETE',OUT/'listen.html',flush=True)


if __name__=='__main__':main()
