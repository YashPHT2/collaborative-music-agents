"""Generate a fixed seed collection from an existing checkpoint; no training."""
import os
os.environ.setdefault('USE_TF','0')
os.environ.setdefault('USE_FLAX','0')
import argparse
import hashlib
import html
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


def page(output,protocol,results):
    cards=[]
    for index,seed in enumerate(protocol['seeds'],1):
        name=f'piece_{index:02d}'
        if name not in results:
            cards.append(f'<section><h2>Study sketch {index}</h2><p>Preparing this piece…</p></section>')
            continue
        cards.append(f'''<section><h2>Study sketch {index}</h2>
<audio controls preload="metadata" src="{name}.mp3"></audio>
<button class="restart">Play from beginning</button>
<div class="downloads"><a download href="{name}.mp3">Download MP3</a>
<a download href="{name}.wav">WAV</a>
<a download href="{name}/orchestra_multiport.mid">MIDI</a></div>
<details><summary>About this sketch</summary><p>Seed {seed} · 8 bars · 70 BPM.
Same A100 phrase checkpoint and sound settings as the version you preferred.</p></details></section>''')
    document='''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><link rel="icon" href="data:,">
<title>More from your A100 model</title><style>
*{box-sizing:border-box}body{font:18px/1.6 system-ui;background:#101d24;color:#edf4f1;max-width:920px;margin:36px auto;padding:20px}
h1{line-height:1.2}section{background:#20323a;padding:24px;border-radius:16px;margin:22px 0}
h2{margin-top:0}audio{width:100%;display:block;margin:18px 0}button{font:inherit;background:#d1f1df;color:#10251b;border:0;border-radius:8px;padding:10px 16px;cursor:pointer}
a{color:#bce7d0}.downloads{display:flex;gap:20px;flex-wrap:wrap;margin:16px 0}details{font-size:15px;color:#c2d1d2}footer{margin:30px 0;color:#b6c9ca}
</style></head><body><h1>More from your A100 model</h1>
<p>Three new 30-second sketches from the checkpoint you preferred. Same instruments
and gentle sound settings, with a different seed for each performance.</p>
<button id="playlist">Play all ready sketches</button>'''+''.join(cards)+'''
<footer><a href="../listen.html">Back to the laptop / A100 comparison</a>
<details><summary>Your earlier favorite</summary><audio controls preload="metadata" src="../trained.mp3"></audio></details>
<p><a href="generation.json">Generation details</a></p></footer>
<script>
const tracks=[...document.querySelectorAll('section audio')];let playlist=false;
document.getElementById('playlist').onclick=()=>{playlist=true;if(tracks.length){tracks[0].currentTime=0;tracks[0].play().catch(()=>{})}};
for(const [i,a] of tracks.entries()){
  a.closest('section').querySelector('.restart').onclick=()=>{playlist=false;a.currentTime=0;a.play().catch(()=>{})};
  a.addEventListener('ended',()=>{if(playlist&&i+1<tracks.length){tracks[i+1].currentTime=0;tracks[i+1].play().catch(()=>{})}else playlist=false});
}
for(const a of document.querySelectorAll('audio'))a.addEventListener('play',()=>{for(const other of document.querySelectorAll('audio'))if(other!==a)other.pause()});
</script></body></html>'''
    (output/'listen.html').write_text(document,encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'outputs/a100_phrase_snapshot_v1/checkpoints/phrases.pt')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/a100_phrase_snapshot_v1/listening/more')
    p.add_argument('--seeds',type=int,nargs='+',default=[9919102,9919103,9919104])
    args=p.parse_args();output=args.output.resolve();checkpoint=args.checkpoint.resolve()
    if (output/'generation.json').exists():raise ValueError('Collection already exists; preserve it and choose a new output folder')
    if len(args.seeds)!=3 or len(set(args.seeds))!=3:p.error('Supply three distinct predetermined seeds')
    output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.manual_seed(9919200)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if saved.get('format_version')!=4 or saved['roles']!=l.ROLES:raise ValueError('Incompatible checkpoint roles/representation')
    protocol=dict(checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  seeds=args.seeds,bars=8,bpm=70,composition_mode=saved['composition_mode'],
                  calm_profile=saved['calm_profile'],new_training=False,
                  sampling='All predetermined seeds retained; no forced notes, rescoring or sample selection.',
                  rendering='Identical felt piano and quiet orchestral rendering recipe as the preferred comparison.',
                  status='loading_model',pieces={})
    write_json(output/'generation.json',protocol);page(output,protocol,{})
    model=nt.load_model();adapters=l.Adapters(model);base_hash=adapters.base_hash()
    if base_hash!=saved['base_sha256']:raise ValueError('Frozen backbone hash mismatch')
    adapters.load(saved['adapters']);planner=l.Planner().to(model.device);planner.load_state_dict(saved['planner']);planner.eval()
    conditions=l.Conditions().to(model.device);conditions.load_state_dict(saved['conditions']);conditions.eval()
    agents={role:oe.Musician().eval() for role in l.ROLES};preview.OUT=output
    results={}
    for index,seed in enumerate(args.seeds,1):
        name=f'piece_{index:02d}';protocol['status']='generating_'+name;write_json(output/'generation.json',protocol)
        print(f'Generating sketch {index}/3; seed {seed}',flush=True)
        episode=l.compose(model,adapters,agents,planner,conditions,8,seed,saved['composition_mode'],saved['calm_profile'])
        directory=output/name;directory.mkdir();write_json(directory/'episode.json',episode)
        raw,families,fade=preview.render(name,episode);del families,fade
        integrated,peak=loudness(raw)
        target=min(-22.,-2.-(peak-integrated)) if np.isfinite(integrated) else None
        gain=target-integrated if target is not None else 0.
        wav=output/(name+'.wav');mp3=output/(name+'.mp3')
        subprocess.run(['ffmpeg','-y','-v','error','-i',str(raw),'-af',f'volume={gain:.9f}dB','-c:a','pcm_s24le',str(wav)],check=True)
        subprocess.run(['ffmpeg','-y','-v','error','-i',str(wav),'-q:a','2',str(mp3)],check=True)
        decoded=directory/'verification_decoded.wav'
        subprocess.run(['ffmpeg','-y','-v','error','-i',str(mp3),str(decoded)],check=True)
        audio,sr=sf.read(decoded);decoded.unlink()
        assert np.isfinite(audio).all() and float(np.max(np.abs(audio)))<1.
        assert sorted(episode['notes'])==sorted(d['note'] for d in episode['decisions'])
        note_hash=hashlib.sha256(json.dumps(episode['notes']).encode()).hexdigest()
        results[name]=dict(seed=seed,seconds=len(audio)/sr,notes=len(episode['notes']),notes_sha256=note_hash,
                           measured_lufs=loudness(wav)[0],decoded_peak=float(np.max(np.abs(audio))),
                           participation=preview.participation(episode['notes']),
                           midi_roundtrip_passed=True,all_notes_match_agent_actions=True)
        del audio
        protocol.update(status='ready_'+name,pieces=results);write_json(output/'generation.json',protocol);page(output,protocol,results)
        print(f'Ready: sketch {index}, {len(episode["notes"])} notes',flush=True)
    assert adapters.base_hash()==base_hash
    protocol.update(status='complete',frozen_backbone_unchanged=True,
                    distinct_note_sequences=len({r['notes_sha256'] for r in results.values()}))
    write_json(output/'generation.json',protocol);page(output,protocol,results)
    print('COLLECTION READY',output/'listen.html',flush=True)


if __name__=='__main__':main()
