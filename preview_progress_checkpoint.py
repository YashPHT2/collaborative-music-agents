"""Audition a downloaded progress snapshot against the user's saved favorite."""
import os
os.environ.setdefault('USE_TF','0')
os.environ.setdefault('USE_FLAX','0')
import argparse
import hashlib
import json
from pathlib import Path
import shutil
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


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def page(output,protocol):
    old=protocol['favorite_reported_step'];step=protocol['progress_step'];val=protocol['latest_validation_step']
    cards=[]
    for name,title in [('favorite',f'Earlier favorite — reported step {old}'),('latest',f'Latest downloaded checkpoint — step {step}')]:
        cards.append(f'''<section><h2>{title}</h2><audio controls preload="metadata" src="{name}.mp3"></audio>
<button>Play from beginning</button><div class="downloads"><a download href="{name}.mp3">Download MP3</a>
<a download href="{name}.wav">WAV</a></div></section>''')
    (output/'listen.html').write_text('''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,"><title>Latest A100 checkpoint</title>
<style>body{font:18px/1.7 system-ui;background:#101d24;color:#edf4f1;max-width:940px;margin:35px auto;padding:24px}h1,h2{line-height:1.25}section{background:#20323a;padding:25px;border-radius:16px;margin:24px 0}audio{width:100%;margin:18px 0}button{font:inherit;background:#d1f1df;color:#10251b;border:0;border-radius:8px;padding:11px 17px;cursor:pointer}a{color:#bce7d0}.downloads{display:flex;gap:24px;margin:18px 0}details{color:#bfcece;font-size:15px}</style>
</head><body><h1>Latest A100 checkpoint vs your earlier favorite</h1>
<p>The same 30-second setup, seed, instruments and loudness balance. These use different saved weights.</p>'''+''.join(cards)+f'''
<details><summary>Checkpoint details</summary><p>The latest weights are from phrase-stage step {step}.
The most recent held-out validation in this file was step {val}; results at that step do not validate the later weights.
This preview does not change the remote training run.</p><p><a href="evaluation.json">Audio and generation checks</a></p></details>
<p><a href="../listen.html">Original laptop comparison</a> · <a href="../long/listen.html">Longer piece from the earlier favorite</a></p>
<script>for(const s of document.querySelectorAll('section')){{const a=s.querySelector('audio');s.querySelector('button').onclick=()=>{{a.currentTime=0;a.play().catch(()=>{{}})}};a.addEventListener('play',()=>{{for(const b of document.querySelectorAll('audio'))if(b!==a)b.pause()}})}};</script>
</body></html>''',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);args=p.parse_args()
    source=args.checkpoint.resolve();saved=torch.load(source,map_location='cpu',weights_only=False)
    state=saved['state'];step=state['step'];stage=state['stage']
    if stage!='phrases':raise ValueError('This comparison is for a phrase-stage snapshot; do not mislabel partner checkpoints')
    archive=ROOT/f'outputs/a100_progress_step{step}_v1';archive.mkdir(parents=True,exist_ok=True)
    snapshot=archive/'progress.pt'
    if snapshot.exists() and sha(snapshot)!=sha(source):raise ValueError('Different progress checkpoint already archived at this step')
    if not snapshot.exists():shutil.copy2(source,snapshot)
    assert sha(snapshot)==sha(source)
    write_json(archive/'training_state.json',state)
    collection=ROOT/'outputs/a100_phrase_snapshot_v1/listening';favorite_preview=collection
    output=collection/f'latest_{step}'
    if (output/'protocol.json').exists():raise ValueError('Preview already exists; keep earlier results')
    output.mkdir(parents=True,exist_ok=True)
    favorite_protocol=json.loads((favorite_preview/'protocol.json').read_text())
    favorite_path=ROOT/'outputs/a100_phrase_snapshot_v1/checkpoints/phrases.pt'
    favorite=torch.load(favorite_path,map_location='cpu',weights_only=False)
    if sha(favorite_path)!=favorite_protocol['source_checkpoints']['trained']:raise ValueError('Favorite weights changed')
    for key in ['format_version','roles','families','base_sha256','calm_profile']:
        if saved[key]!=favorite[key]:raise ValueError('Comparison settings mismatch: '+key)
    if saved.get('format_version')!=4 or saved['roles']!=l.ROLES:raise ValueError('Unsupported checkpoint')
    favorite_raw=favorite_preview/'trained/mix_raw.wav'
    if not favorite_raw.is_file():raise ValueError('Original favorite render is unavailable; regenerate the reference explicitly')
    validations=[v for v in state['validations'] if v['stage']==stage]
    last=max(validations,key=lambda v:v['step']) if validations else None
    protocol=dict(seed=favorite_protocol['seed'],bars=8,bpm=70,favorite_reported_step=favorite_protocol['log_selected_step'],
        progress_step=step,progress_stage=stage,training_complete=state['complete'],
        latest_validation_step=last['step'] if last else None,latest_validation_passed=last['passed'] if last else None,
        progress_step_has_validation=any(v['step']==step for v in validations),
        selected_best_step=state['selected_step'],checkpoint_sha256={'favorite':sha(favorite_path),'latest':sha(snapshot)},
        cached_favorite_raw_sha256=sha(favorite_raw),calm_profile=saved['calm_profile'],new_training=False,
        status='generating_latest')
    write_json(output/'protocol.json',protocol)
    torch.set_num_threads(4);torch.manual_seed(protocol['seed'])
    model=nt.load_model();adapters=l.Adapters(model);base_hash=adapters.base_hash()
    if base_hash!=saved['base_sha256']:raise ValueError('Backbone hash mismatch')
    adapters.load(saved['adapters']);planner=l.Planner().to(model.device);planner.load_state_dict(saved['planner']);planner.eval()
    conditions=l.Conditions().to(model.device);conditions.load_state_dict(saved['conditions']);conditions.eval()
    agents={r:oe.Musician().eval() for r in l.ROLES}
    def progress(done,notes):print(f'Latest checkpoint: {done}/8 bars; {notes} agent notes',flush=True)
    episode=l.compose(model,adapters,agents,planner,conditions,8,protocol['seed'],saved['composition_mode'],saved['calm_profile'],progress=progress)
    write_json(output/'latest_episode.json',episode);preview.OUT=output
    latest_raw,families,fade=preview.render('latest',episode);del families,fade
    renders={'favorite':favorite_raw,'latest':latest_raw};levels={name:loudness(path) for name,path in renders.items()}
    finite=[(integrated,peak) for integrated,peak in levels.values() if np.isfinite(integrated)]
    target=min([-22.]+[-2.-(peak-integrated) for integrated,peak in finite]);checks={}
    for name,raw in renders.items():
        gain=target-levels[name][0] if np.isfinite(levels[name][0]) else 0.
        wav=output/(name+'.wav');mp3=output/(name+'.mp3')
        subprocess.run(['ffmpeg','-y','-v','error','-i',str(raw),'-af',f'volume={gain:.9f}dB','-c:a','pcm_s24le',str(wav)],check=True)
        subprocess.run(['ffmpeg','-y','-v','error','-i',str(wav),'-q:a','2',str(mp3)],check=True)
        decoded=output/(name+'_verification.wav');subprocess.run(['ffmpeg','-y','-v','error','-i',str(mp3),str(decoded)],check=True)
        audio,sr=sf.read(decoded);decoded.unlink();assert np.isfinite(audio).all() and float(np.max(np.abs(audio)))<1.
        notes=episode['notes'] if name=='latest' else json.loads((favorite_preview/'trained_episode.json').read_text())['notes']
        checks[name]=dict(seconds=len(audio)/sr,decoded_peak=float(np.max(np.abs(audio))),measured_lufs=loudness(wav)[0],
                          notes=len(notes),participation=preview.participation(notes))
        del audio
    assert sorted(episode['notes'])==sorted(d['note'] for d in episode['decisions']) and adapters.base_hash()==base_hash
    write_json(output/'evaluation.json',dict(audio=checks,all_notes_match_agent_actions=True,midi_roundtrip_passed=True,
        frozen_backbone_unchanged=True,comparison_seed=protocol['seed'],checkpoint_metadata=protocol,
        listening_preference='unknown',cooperation_established=False))
    protocol['status']='complete';write_json(output/'protocol.json',protocol);page(output,protocol)
    print('LATEST CHECKPOINT COMPARISON READY',output/'listen.html',flush=True)


if __name__=='__main__':main()
