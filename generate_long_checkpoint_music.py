"""Continuous extended generation from the preferred saved A100 phrase weights."""
import os
os.environ.setdefault('USE_TF','0')
os.environ.setdefault('USE_FLAX','0')
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import soundfile as sf
import torch

import evaluate_trained_orchestra as preview
import orchestra_learning as l
import note_trio as nt
import offline_ensemble as oe
from audit_projected_roles import loudness
from train_ensemble_repair import write_json

ROOT=Path(__file__).resolve().parent


def page(output,protocol):
    (output/'listen.html').write_text('''<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="icon" href="data:,"><title>A longer piece from your A100 model</title>
<style>body{font:18px/1.7 system-ui;background:#101d24;color:#edf4f1;max-width:900px;margin:50px auto;padding:24px}h1{line-height:1.25}section{background:#20323a;padding:28px;border-radius:18px;margin:30px 0}audio{width:100%;margin:22px 0}button{font:inherit;background:#d1f1df;color:#10251b;border:0;border-radius:8px;padding:12px 18px;cursor:pointer}a{color:#bce7d0}.downloads{display:flex;gap:24px;flex-wrap:wrap;margin:24px 0}details{font-size:15px;color:#bfd0d0}</style>
</head><body><h1>A longer piece from your A100 model</h1>
<p>One continuous performance, about 1 minute 53 seconds, from the checkpoint you preferred.</p>
<section><audio id="player" controls preload="metadata" src="long_piece.mp3"></audio>
<button id="restart">Play from beginning</button><div class="downloads">
<a download href="long_piece.mp3">Download MP3</a><a download href="long_piece.wav">WAV</a>
<a download href="performance/orchestra_multiport.mid">MIDI</a></div>
<details><summary>About this performance</summary><p>32 bars at 70 BPM. The agents generate notes continuously while retaining recent ensemble history. Same felt piano and quiet orchestral sounds as the shorter clips. This extends generation from a model trained on shorter phrases; long-form development has not been separately trained.</p></details></section>
<p><a href="../more/listen.html">Back to the three short sketches</a> · <a href="../listen.html">Original comparison</a></p>
<details><summary>Generation details</summary><a href="generation.json">Open diagnostics</a></details>
<script>const a=document.getElementById('player');document.getElementById('restart').onclick=()=>{a.currentTime=0;a.play().catch(()=>{})};</script>
</body></html>''',encoding='utf-8')
    if protocol.get('checkpoint_step') is not None:
        path=output/'listen.html';markup=path.read_text(encoding='utf-8');step=protocol['checkpoint_step']
        markup=markup.replace('A longer piece from your A100 model',f'A long piece from A100 step {step}')
        markup=markup.replace('from the checkpoint you preferred.',f'from the step-{step} checkpoint.')
        markup=markup.replace('<p><a href="../more/listen.html">','<p><a href="../long/listen.html">Earlier long piece</a> · <a href="../more/listen.html">')
        path.write_text(markup,encoding='utf-8')
    if protocol.get('profile')=='energetic':
        path=output/'listen.html';markup=path.read_text(encoding='utf-8')
        markup=markup.replace(f'A long piece from A100 step {protocol["checkpoint_step"]}',f'Fast-paced A100 — {protocol["bpm"]} BPM')
        markup=markup.replace('A longer piece from your A100 model',f'Fast-paced A100 — {protocol["bpm"]} BPM')
        markup=markup.replace('One continuous performance, about 1 minute 53 seconds, from the step-'+str(protocol['checkpoint_step'])+' checkpoint.',
            f'A fresh 30-second performance at {protocol["bpm"]} BPM, from the step-{protocol["checkpoint_step"]} checkpoint.')
        markup=markup.replace('32 bars at 70 BPM.',f'{protocol["bars"]} bars at {protocol["bpm"]} BPM, with denser activity targets and a broader dynamics range.')
        markup=markup.replace('long_piece','fast_piece')
        path.write_text(markup,encoding='utf-8')
    if protocol.get('profile')=='cinematic':
        path=output/'listen.html';markup=path.read_text(encoding='utf-8')
        markup=markup.replace(f'A long piece from A100 step {protocol["checkpoint_step"]}','Original cinematic organ sketch')
        markup=markup.replace('A longer piece from your A100 model','Original cinematic organ sketch')
        markup=markup.replace('One continuous performance, about 1 minute 53 seconds, from the step-'+str(protocol['checkpoint_step'])+' checkpoint.',
            'An original one-minute organ-led performance, with piano motion and a supporting orchestral texture. This is new music from the agents, rather than a recreation of the Interstellar theme.')
        markup=markup.replace('32 bars at 70 BPM.',f'{protocol["bars"]} bars at {protocol["bpm"]} BPM.')
        markup=markup.replace('Same felt piano and quiet orchestral sounds as the shorter clips.',
            'An explicit organ-led mix and cinematic planning controls guide this generation. No reference audio or transcribed soundtrack notes were used.')
        markup=markup.replace('long_piece','cinematic_piece')
        path.write_text(markup,encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'outputs/a100_phrase_snapshot_v1/checkpoints/phrases.pt')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/a100_phrase_snapshot_v1/listening/long')
    p.add_argument('--seed',type=int,default=9919110)
    p.add_argument('--bars',type=int,default=32);p.add_argument('--bpm',type=int,default=70)
    p.add_argument('--seconds',type=float);p.add_argument('--profile',choices=['calm','energetic','cinematic'],default='calm')
    args=p.parse_args();out=args.output.resolve();checkpoint=args.checkpoint.resolve()
    if (out/'generation.json').exists():raise ValueError('Output already exists; preserve the earlier performance')
    bars=args.bars;bpm=args.bpm;beats=bars*4
    if not 1<=bars<=32 or not 40<=bpm<=180:p.error('Use 1..32 bars and 40..180 BPM')
    music_seconds=beats*round(60_000_000/bpm)/1_000_000
    seconds=args.seconds if args.seconds is not None else music_seconds+3
    if not music_seconds<=seconds<=music_seconds+10:p.error('Duration must include the full score and no more than 10 seconds of tail')
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(args.seed)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if saved.get('format_version')!=4 or saved['roles']!=l.ROLES:raise ValueError('Incompatible checkpoint')
    controls=copy.deepcopy(saved['calm_profile'])
    if args.profile=='energetic':
        controls['density']=[max(value,.5 if s['family']=='pianos' else .3) for value,s in zip(controls['density'],l.ROLES.values())]
        controls['velocity']=[.55 if s['family']=='pianos' else .48 for s in l.ROLES.values()]
        controls['style']=[.4,.62,.18,controls['style'][3]]
        controls['velocity_max']=76
    if args.profile=='cinematic':
        density=dict(organ=.65,pianos=.4,strings=.3,woodwinds=.15,brass=.15,synth=.18,choir=.18)
        dynamics=dict(organ=.56,pianos=.44,strings=.47,woodwinds=.38,brass=.38,synth=.35,choir=.35)
        controls['density']=[density[s['family']] for s in l.ROLES.values()]
        controls['velocity']=[dynamics[s['family']] for s in l.ROLES.values()]
        controls['style']=[.3,.56,.3,1.]
        controls['velocity_max']=76
    protocol=dict(status='loading_model',checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        checkpoint_step=saved.get('state',{}).get('step'),checkpoint_stage=saved.get('state',{}).get('stage'),
        bars=bars,bpm=bpm,seed=args.seed,mode=saved['composition_mode'],calm_profile=controls,profile=args.profile,
        new_training=False,continuous_generation=True,recent_history_beats=24,
        phrase_planning_interval_bars=2,joined_audio_clips=False,forced_notes=False,
        mix_profile='organ_lead' if args.profile=='cinematic' else 'focus',
        scope='Extended inference from four-bar supervised targets; long-form development has not been separately trained.')
    write_json(out/'generation.json',protocol)
    model=nt.load_model();adapters=l.Adapters(model);base_hash=adapters.base_hash()
    if base_hash!=saved['base_sha256']:raise ValueError('Frozen backbone mismatch')
    adapters.load(saved['adapters']);planner=l.Planner().to(model.device);planner.load_state_dict(saved['planner']);planner.eval()
    conditions=l.Conditions().to(model.device);conditions.load_state_dict(saved['conditions']);conditions.eval()
    agents={role:oe.Musician().eval() for role in l.ROLES}
    def progress(done,notes):
        protocol.update(status='generating',completed_bars=done,note_count=notes);write_json(out/'generation.json',protocol)
        print(f'Generated {done}/{bars} bars; {notes} agent notes',flush=True)
    episode=l.compose(model,adapters,agents,planner,conditions,bars,args.seed,saved['composition_mode'],controls,progress=progress,bpm=bpm)
    write_json(out/'episode.json',episode)
    assert adapters.base_hash()==base_hash
    protocol.update(status='rendering',frozen_backbone_unchanged=True);write_json(out/'generation.json',protocol)
    preview.OUT=out;preview.BARS=bars;preview.BPM=bpm;preview.SECONDS=seconds
    raw,families,fade=preview.render('performance',episode,mix_profile=protocol['mix_profile']);del families,fade
    integrated,peak=loudness(raw)
    target=min(-22.,-2.-(peak-integrated)) if np.isfinite(integrated) else None
    gain=target-integrated if target is not None else 0.
    stem={'energetic':'fast_piece','cinematic':'cinematic_piece','calm':'long_piece'}[args.profile]
    wav=out/(stem+'.wav');mp3=out/(stem+'.mp3')
    subprocess.run(['ffmpeg','-y','-v','error','-i',str(raw),'-af',f'volume={gain:.9f}dB','-c:a','pcm_s24le',str(wav)],check=True)
    subprocess.run(['ffmpeg','-y','-v','error','-i',str(wav),'-q:a','2',str(mp3)],check=True)
    decoded=out/'verification_decoded.wav';subprocess.run(['ffmpeg','-y','-v','error','-i',str(mp3),str(decoded)],check=True)
    audio,sr=sf.read(decoded);decoded.unlink()
    assert np.isfinite(audio).all() and float(np.max(np.abs(audio)))<1.
    assert sorted(episode['notes'])==sorted(d['note'] for d in episode['decisions'])
    active=np.zeros(beats*16,dtype=bool)
    for n in episode['notes']:active[n[1]//30:min(len(active),(n[1]+n[2])//30)]=True
    longest=run=0
    for sounding in active:run=0 if sounding else run+1;longest=max(longest,run)
    protocol.update(status='complete',seconds=len(audio)/sr,decoded_peak=float(np.max(np.abs(audio))),
        measured_lufs=loudness(wav)[0],all_notes_match_agent_actions=True,midi_roundtrip_passed=True,
        note_count=len(episode['notes']),participation=preview.participation(episode['notes'],beats),
        ensemble_sounding_beats=float(active.sum()/16),longest_ensemble_silence_beats=longest/16,
        last_note_end_beat=max((n[1]+n[2] for n in episode['notes']),default=0)/480)
    write_json(out/'generation.json',protocol);page(out,protocol)
    print('LONG PIECE READY',out/'listen.html',flush=True)


if __name__=='__main__':main()
