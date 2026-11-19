"""Matched blind listening and held-out partner interventions; no training."""
import copy
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import ensemble_repair as er
import note_trio as nt
import offline_ensemble as oe
import supervised_ensemble as se
import supported_ensemble_loader as loader
from audit_projected_roles import loudness
from generate_piano import render
from train_ensemble_repair import write_json

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'outputs/supported_ensemble_run_v1'
OUT=ROOT/'outputs/supported_comparison_v1'
SEED=9899101
MODELS={'earlier':ROOT/'outputs/offline_ensemble_v1/revised.pt',
        'phrase':RUN/'participation.pt','partner':RUN/'selected.pt'}
CONTEXTS=('correct','missing','mismatched')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def intervene(notes,role,context):
    notes=copy.deepcopy(notes);channel=er.ROLES.index(role)
    if context=='missing':return [n for n in notes if n[3]==channel]
    if context=='mismatched':
        for n in notes:
            if n[3]!=channel:n[4]=min(127,n[4]+6)
    return notes


def response_metrics(ledger,made,role):
    """Describe only the target response and its overlap with actual partners."""
    channel=er.ROLES.index(role);left,right=3840,7680
    own=[n for n in ledger if n[3]==channel and n[1]<right and n[1]+n[2]>left]
    partners=[n for n in ledger if n[3]!=channel and n[1]<right and n[1]+n[2]>left]
    consonant=overlap=0
    for a in own:
        for b in partners:
            ticks=max(0,min(right,a[1]+a[2],b[1]+b[2])-max(left,a[1],b[1]))
            overlap+=ticks;consonant+=ticks*(abs(a[4]-b[4])%12 in (0,3,4,5,7,8,9))
    occupied=np.array([any(n[1]<=t<n[1]+n[2] for n in own) for t in range(left,right,30)])
    return dict(generated_notes=len(made),sounding_beats=float(occupied.sum()/16),
                onset_density=len(made)/8,overlapping_note_pair_beats=overlap/480,
                consonant_overlap_fraction=consonant/overlap if overlap else None)


@torch.no_grad()
def predictive_probe(model,adapters,planner,conditions,w,role,context):
    """True own targets stay fixed; only observed partner information changes."""
    changed=intervene(w['notes'],role,context)
    plans=er.plan_sequence(planner,changed,w['bars']);examples=[];deltas=[];beats=[]
    actual=[n for n in w['notes'] if n[3]!=er.ROLES.index(role) and 3840<=n[1]<7680]
    announced=intervene(actual,role,context)
    features=oe.draft_features(announced,role,3840).to(model.device)
    adapters.activate(role)
    with oe.limits():
        for beat in range(8,16):
            history=intervene(nt.visible(w['notes'],beat),role,context)
            ex=se.teacher_example(model,role,{**w,'beat':beat,'sampler_type':oe.Sampler},history=history)
            if ex is None:continue
            examples.append(ex);deltas.append(conditions(plans[beat//4],features,role));beats.append(beat)
        values=er.batch_category_losses(model,examples,deltas)
    return er.aggregate([dict(group=w['group'],kind=er.beat_kind(w,role,beat),losses={k:float(v) for k,v in losses.items()})
                         for beat,losses in zip(beats,values)])


def bootstrap(values):
    if not values:return None
    rng=random.Random(9899701)
    draws=sorted(float(np.mean(rng.choices(values,k=len(values)))) for _ in range(2000))
    return [draws[49],draws[1949]]


def summarize(cases):
    result={}
    for role in er.ROLES:
        rows=[r for r in cases if r['role']==role];pred={};generated={}
        for context in ('missing','mismatched'):
            pred[context]={}
            for metric in ('event','attributes'):
                gains=[r['predictive'][context][metric]-r['predictive']['correct'][metric] for r in rows
                       if r['predictive'][context][metric] is not None and r['predictive']['correct'][metric] is not None]
                pred[context][metric]=dict(mean_loss_reduction_with_correct=float(np.mean(gains)) if gains else None,
                                           bootstrap_95_interval=bootstrap(gains))
            pairs=[r['response']['correct']['consonant_overlap_fraction']-r['response'][context]['consonant_overlap_fraction']
                   for r in rows if all(r['response'][c]['consonant_overlap_fraction'] is not None for c in ('correct',context))]
            generated[context]=dict(comparable_groups=len(pairs),mean_consonance_advantage=float(np.mean(pairs)) if pairs else None,
                                    bootstrap_95_interval=bootstrap(pairs),
                                    changed_outputs=sum(r['generated']['correct']!=r['generated'][context] for r in rows))
        baseline_pairs=[r['response']['correct']['consonant_overlap_fraction']-r['earlier_correct']['consonant_overlap_fraction']
                        for r in rows if r['response']['correct']['consonant_overlap_fraction'] is not None
                        and r['earlier_correct']['consonant_overlap_fraction'] is not None]
        result[role]=dict(compositions=len(rows),predictive_context_advantage=pred,response_context_diagnostics=generated,
                         matched_schedule_change_from_earlier=dict(comparable_groups=len(baseline_pairs),
                            mean_consonance_change=float(np.mean(baseline_pairs)) if baseline_pairs else None,
                            bootstrap_95_interval=bootstrap(baseline_pairs)),
                         cooperation_established=False,
                         limitation='Small held-out set; consonance is a low-level diagnostic, not a validated taste or cooperation reward.')
    return result


def render_panel(episodes,mapping):
    checks={label:{} for label in mapping};duration=21.6
    for stem in ('mix',*er.ROLES):
        raw={}
        for label,variant in mapping.items():
            ep=episodes[variant];base=OUT/f'{label}_{stem}';midi=base.with_suffix('.mid')
            nt.midi_file(midi,ep['notes'],None if stem=='mix' else stem,bpm=100)
            wav=base.with_suffix('.wav');render(midi,wav);trim=OUT/f'{label}_{stem}_trim.wav'
            subprocess.run(['ffmpeg','-y','-v','error','-i',str(wav),'-af',
                f'apad,atrim=duration={duration},afade=t=out:st={duration-1}:d=1','-c:a','pcm_s24le',str(trim)],check=True)
            data,_=sf.read(trim);silent=float(np.max(np.abs(data)))<1e-7
            integrated,peak=(None,None) if silent else loudness(trim)
            raw[label]=dict(path=trim,silent=silent,integrated=integrated,peak=peak)
        target=min([-18 if stem=='mix' else -22]+[-2-(v['peak']-v['integrated']) for v in raw.values() if not v['silent']])
        for label,measure in raw.items():
            base=OUT/f'{label}_{stem}';wav=base.with_suffix('.wav');mp3=base.with_suffix('.mp3')
            gain=0 if measure['silent'] else target-measure['integrated']
            subprocess.run(['ffmpeg','-y','-v','error','-i',str(measure['path']),'-af',f'volume={gain:.9f}dB','-c:a','pcm_s24le',str(wav)],check=True)
            subprocess.run(['ffmpeg','-y','-v','error','-i',str(wav),'-q:a','2',str(mp3)],check=True)
            data,sr=sf.read(wav)
            if not np.isfinite(data).all() or np.max(np.abs(data))>=1:raise ValueError('Nonfinite/clipped audio')
            decoded=OUT/f'{label}_{stem}_decoded.wav'
            subprocess.run(['ffmpeg','-y','-v','error','-i',str(mp3),str(decoded)],check=True)
            decoded_audio,_=sf.read(decoded);decoded.unlink()
            if not np.isfinite(decoded_audio).all() or np.max(np.abs(decoded_audio))>=1:raise ValueError('MP3 decoded audio clipped')
            checks[label][stem]=dict(seconds=len(data)/sr,silent=measure['silent'],constant_gain_db=gain,
                target_lufs=None if measure['silent'] else target,
                measured_lufs=None if measure['silent'] else loudness(wav)[0],sample_peak=float(np.max(np.abs(data))),
                mp3_decoded_peak=float(np.max(np.abs(decoded_audio))),sha256=sha(mp3),midi_action_agreement=True)
            measure['path'].unlink()
        print('Rendered matched mixes/stems:',stem,flush=True)
    write_json(OUT/'audio_verification.json',checks)


def page():
    (OUT/'listen.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,"><title>Three trio clips — blind comparison</title>
<style>body{font:18px system-ui;background:#101724;color:#edf2fa;max-width:950px;margin:30px auto;padding:20px;line-height:1.55}section{background:#1d2b40;padding:22px;border-radius:14px;margin:22px 0}audio{width:100%;margin:16px 0}select,input,button{font:inherit;padding:9px;border-radius:7px;max-width:100%}label{display:block;margin:12px 0}.ratings{display:grid;grid-template-columns:1fr 1fr;gap:8px}a{color:#9dd8ff}.muted{color:#b8c5d8}@media(max-width:600px){.ratings{grid-template-columns:1fr}}</style>
<h1>Which trio sounds better?</h1><p>Three newly generated performances, with the same seed, eight bars, tempo and sound library. Every note comes from an agent. The three mixes have matched loudness. Model identities are hidden until you reveal them.</p><p>Listen to all three <strong>full trios</strong> first. Rate each independently; use solos to investigate a weak part. Silence is kept if an agent chooses it. Scores do not need to be different.</p>
<label>Name or nickname <input id="voter" value="Anonymous"></label><main id="clips"></main>
<button id="download">Download my ratings</button> <button id="reveal">Reveal models</button><p id="message" role="status"></p><pre id="identities"></pre><p id="probe-status"></p><p><a href="REPORT.md">Methods and limitations</a> · <a href="evaluation.json">Technical evaluation</a></p>
<script>
const labels=['A','B','C'],dimensions={enjoyment:'Enjoyment',development:'Musical development',ensemble_fit:'Parts fitting together',warmth:'Warmth / hopefulness'},storage='supported-comparison-v1-9899101';let saved={};try{saved=JSON.parse(localStorage.getItem(storage)||'{}')}catch{}
for(const label of labels){const card=document.createElement('section');card.dataset.clip=label;card.innerHTML=`<h2>Clip ${label}</h2><label>Part <select class="stem"><option value="mix">Full trio</option><option value="organ">Organ solo</option><option value="strings">Strings solo</option><option value="cello">Cello solo</option></select></label><audio controls preload="metadata" src="${label}_mix.mp3"></audio><button class="restart">Play from beginning</button><p class="part-note muted"></p><div class="ratings">${Object.entries(dimensions).map(([key,title])=>`<label>${title} <select data-rating="${key}"><option value="">Unrated</option>${[1,2,3,4,5].map(n=>`<option value="${n}">${n} / 5</option>`).join('')}</select></label>`).join('')}</div>`;document.getElementById('clips').append(card);const audio=card.querySelector('audio');card.querySelector('.stem').onchange=e=>{audio.pause();audio.src=`${label}_${e.target.value}.mp3`;audio.load();fetch('audio_verification.json').then(r=>r.json()).then(v=>card.querySelector('.part-note').textContent=v[label][e.target.value].silent?'This part is silent in this performance.':'').catch(()=>{})};card.querySelector('.restart').onclick=()=>{audio.currentTime=0;audio.play().catch(()=>document.getElementById('message').textContent='Press the audio play button.')};for(const select of card.querySelectorAll('[data-rating]')){select.value=saved[label]?.[select.dataset.rating]||'';select.onchange=()=>{(saved[label]??={})[select.dataset.rating]=select.value;localStorage.setItem(storage,JSON.stringify(saved));document.getElementById('message').textContent='Ratings saved in this browser.'}}}
for(const a of document.querySelectorAll('audio'))a.addEventListener('play',()=>{for(const other of document.querySelectorAll('audio'))if(other!==a)other.pause()});
document.getElementById('download').onclick=()=>{if(labels.some(l=>Object.keys(dimensions).some(k=>!saved[l]?.[k]))){document.getElementById('message').textContent='Please rate all four questions for each clip.';return}const data={schema:'supported-comparison-v1',seed:9899101,voter:document.getElementById('voter').value,identities_revealed:localStorage.getItem(storage+'-revealed')==='true'||!!document.getElementById('identities').textContent,ratings:saved};const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download='trio-comparison-votes.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
document.getElementById('reveal').onclick=async()=>{const key=await(await fetch('panel_key.json')).json();localStorage.setItem(storage+'-revealed','true');document.getElementById('identities').textContent=Object.entries(key.labels).map(([label,value])=>label+': '+value.description).join(String.fromCharCode(10))};
async function status(){try{const e=await(await fetch('evaluation.json',{cache:'no-store'})).json();document.getElementById('probe-status').textContent=e.status==='complete'?'Partner tests finished. These are technical diagnostics; listener preference is still unknown.':'Partner tests are running; these audio clips are ready.'}catch{}setTimeout(status,10000)}status();
</script></html>''',encoding='utf-8')


def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.manual_seed(SEED)
    state=json.loads((RUN/'training_log.json').read_text())
    if not state['complete']:raise ValueError('Training is not complete')
    examples,groups,manifest=loader.load_data()
    probes=[]
    for role in er.ROLES:
        for group,windows in sorted(groups['test'][role].items()):
            candidates=sorted([w for w in windows if w['loss_start_beat']==8 and len(w['available_roles'])==3],key=lambda w:w['id'])
            if candidates:probes.append(dict(role=role,group=group,window_id=candidates[0]['id'],seed=9899800+len(probes)))
    variants=list(MODELS);random.Random(9899007).shuffle(variants);mapping=dict(zip('ABC',variants))
    protocol=dict(seed=SEED,bars=8,bpm=100,seconds_without_tail=19.2,profile=None,prefix=None,
                  data_sha256=manifest['eligible_examples_sha256'],checkpoints={k:dict(path=str(p),sha256=sha(p)) for k,p in MODELS.items()},
                  selection=state['selections'],probe_cases=probes,contexts=CONTEXTS,
                  probes='All available untouched test groups with a legal two-bar prefix. Identical seed and actual partners across contexts; mismatched observations transpose partners +6 semitones.',
                  quality='Unmeasured until listener ratings; no reward or training is changed')
    if (OUT/'protocol.json').exists() and json.loads((OUT/'protocol.json').read_text())!=json.loads(json.dumps(protocol)):raise ValueError('Protocol changed')
    write_json(OUT/'protocol.json',protocol)
    descriptions={'earlier':'Earlier model (original v2 revised generation)',
                  'phrase':'New phrase/participation model (selected update 128)',
                  'partner':'New partner-response model (selected update 80)'}
    write_json(OUT/'panel_key.json',dict(labels={k:dict(variant=v,description=descriptions[v]) for k,v in mapping.items()}))
    write_json(OUT/'evaluation.json',dict(status='generating',listener_preference='unknown',cooperation_established=False))
    model,adapters,agents,planner,conditions=er.load(MODELS['partner']);base_hash=adapters.base_hash()
    def restore(variant):
        saved=torch.load(MODELS[variant],map_location='cpu',weights_only=False);adapters.load(saved['adapters'])
        if saved['format_version']==2:
            p,c=er.upgrade(saved,model.device);planner.load_state_dict(p.state_dict());conditions.load_state_dict(c.state_dict())
        else:planner.load_state_dict(saved['planner']);conditions.load_state_dict(saved['conditions'])
        planner.eval();conditions.eval();return saved
    episodes={};metrics={}
    for variant in variants:
        saved=restore(variant);artifact=OUT/f'{variant}_episode.json'
        if artifact.exists():ep=json.loads(artifact.read_text())
        elif variant=='earlier':
            oldp=oe.Planner().to(model.device);oldp.load_state_dict(saved['planner']);oldp.eval()
            oldc=oe.Conditions().to(model.device);oldc.load_state_dict(saved['conditions']);oldc.eval()
            ep=oe.compose(model,adapters,agents,oldp,oldc,8,SEED,'revised');write_json(artifact,ep)
        else:
            ep=er.compose(model,adapters,agents,planner,conditions,8,SEED,saved['composition_mode']);write_json(artifact,ep)
        assert not ep['dataset_prefix'] and sorted(ep['notes'])==sorted(d['note'] for d in ep['decisions'])
        ledger=[]
        with oe.limits():
            for beat in range(32):ledger=nt.commit(ledger,{r:[n for n in ep['notes'] if n[3]==i and n[1]//480==beat] for i,r in enumerate(er.ROLES)},beat,nt.Brief(8))
        assert ledger==ep['notes']
        episodes[variant]=ep;metrics[variant]=er.participation(ep['notes'])
        print('Generated fixed-seed clip:',variant,{r:v['notes'] for r,v in metrics[variant].items()},flush=True)
    render_panel(episodes,mapping)
    write_json(OUT/'participation.json',dict(generated=metrics,reference_test=[dict(group=w['group'],id=w['id'],roles=w['roles']) for w in examples if w['split']=='test']))
    (OUT/'REPORT.md').write_text('''# Matched trio comparison

The earlier v2 model, the selected participation checkpoint and the selected
partner-response checkpoint each generate eight bars from an empty ledger,
using the same predetermined seed 9899101, 100 BPM, role constraints and patches.
There is no supplied composition prefix, copied motif, warm-build profile or
hand-added note. Every exported note must match a committed agent action.
The earlier model uses its original v2 planner schedule; the two new models use
their production v3 schedule. This compares complete systems, not an isolated
weight-only ablation. The fixed random A/B/C mapping is saved in panel_key.json.

All three full mixes share one integrated loudness target and identical FluidSynth
settings. Each set of three solos is similarly matched. Only constant gain, fixed
length padding and a tail fade are applied. Rests remain unchanged. MIDI round
trips, WAV/decoded MP3 peaks and nonfinite audio are checked.

Partner probes use every eligible untouched test group with a two-bar prefix,
one deterministic window per composition and role. Correct, missing and +6-semitone
mismatched observations use identical seeds and actual, unchanged partner parts.
Only the learner's notes vary. Prefix sustains count toward sounding coverage;
zero overlap is reported as unavailable consonance rather than rewarded silence.
The earlier checkpoint is also probed with correct context after a v3 schedule
upgrade to provide a matched-schedule baseline, distinct from its listening clip.

Predictive event/attribute losses use fixed real target notes and interventions on
partner observations; announced partner drafts are oracle real parts in this test.
They measure sensitivity/usefulness under this controlled condition, not free-running
cooperation. Generated consonant-overlap fractions are unvalidated diagnostics.
Small per-role group counts and bootstrap intervals are reported. No pass on these
metrics establishes musical enjoyment, and no RL reward is enabled by this run.

The protocol and all probe outputs are recorded before summarization; there is no
selection of attractive clips or best seeds. Listening preference remains unknown.
''',encoding='utf-8')
    write_json(OUT/'evaluation.json',dict(status='probes_running',listener_preference='unknown',cooperation_established=False))
    page();print('Listening clips ready:',OUT/'listen.html',flush=True)
    # Production execution-order invariant, using the selected partner model.
    restore('partner')
    normal=er.compose(model,adapters,agents,planner,conditions,2,9899700,'revised')
    reverse=er.compose(model,adapters,agents,planner,conditions,2,9899700,'revised',order=list(reversed(er.ROLES)))
    if normal['notes']!=reverse['notes']:raise ValueError('Agent order changed joint output')
    cases_path=OUT/'partner_cases.json';cases=json.loads(cases_path.read_text()) if cases_path.exists() else []
    by_id={w['id']:w for w in examples}
    for index,case in enumerate(probes):
        if any(r['role']==case['role'] and r['group']==case['group'] for r in cases):continue
        w=by_id[case['window_id']];role=case['role'];restore('earlier')
        baseline_ledger,baseline_notes=er.fixed_response(model,adapters,agents,planner,conditions,w,role,case['seed'],'correct')
        result={**case,'earlier_correct':response_metrics(baseline_ledger,baseline_notes,role),
                'response':{},'generated':{},'predictive':{},'actual_partner_parts_unchanged':True}
        restore('partner')
        for context in CONTEXTS:
            ledger,made=er.fixed_response(model,adapters,agents,planner,conditions,w,role,case['seed'],context)
            result['response'][context]=response_metrics(ledger,made,role);result['generated'][context]=made
            result['predictive'][context]=predictive_probe(model,adapters,planner,conditions,w,role,context)
        cases.append(result);write_json(cases_path,cases)
        write_json(OUT/'evaluation.json',dict(status='probes_running',completed=len(cases),total=len(probes),per_role=summarize(cases),listener_preference='unknown',cooperation_established=False))
        print(f'Partner case {index+1}/{len(probes)}: {role} {case["group"]}',flush=True)
    if adapters.base_hash()!=base_hash:raise ValueError('Frozen backbone changed')
    if any(sha(p)!=protocol['checkpoints'][v]['sha256'] for v,p in MODELS.items()):raise ValueError('Checkpoint changed')
    write_json(OUT/'integration_checks.json',dict(agent_order_independent=True,frozen_backbone_unchanged=True,checkpoints_unchanged=True,
               midi_action_agreement=True,actual_probe_partner_parts_unchanged=True,no_prefix_in_listening_clips=True))
    write_json(OUT/'evaluation.json',dict(status='complete',per_role=summarize(cases),listener_preference='unknown',
               cooperation_established=False,completed=len(cases),total=len(probes)))
    print('Comparison and partner diagnostics complete',flush=True)


if __name__=='__main__':main()
