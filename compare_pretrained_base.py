"""Untouched pretrained backbone versus existing Clip B under matched sampling."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import soundfile as sf
import torch

import note_trio as nt
import offline_ensemble as oe
from band import fingerprint
from generate_piano import render
from audit_projected_roles import loudness
from train_ensemble_repair import write_json

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'outputs/supported_comparison_v1'
OUT=ROOT/'outputs/pretrained_vs_clip_b_v1'
SEED=9899101


def adapter_compatible_base_hash(model):
    # Adapter wrapping renames q/v weights to base_layer without changing them.
    # Normalize names only so this raw model can be checked against that digest.
    digest=hashlib.sha256()
    for name,value in model.state_dict().items():
        for projection in ('q_proj','v_proj'):
            name=name.replace('.'+projection+'.','.'+projection+'.base_layer.')
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def compose(model,agents,bars=8,seed=SEED,order=None):
    """Same private two-bar proposals and synchronous commits as Clip B; no delta."""
    brief=nt.Brief(bars);ledger=[];decisions=[];trace=[]
    with oe.limits():
        for start in range(0,bars*4,8):
            proposals={}
            for role in order or oe.ROLES:
                own=copy.deepcopy(ledger);made=[]
                for beat in range(start,min(start+8,bars*4)):
                    history=nt.visible(own,beat)
                    sampler=oe.Sampler(model,role,seed+beat*1009+oe.ROLES.index(role)*100003,brief,beat,history)
                    notes,_=sampler.propose(agents[role],nt.state_features(brief,beat,history))
                    own=nt.commit(own,{r:notes if r==role else [] for r in oe.ROLES},beat,brief);made.extend(notes)
                proposals[role]=made
            for beat in range(start,min(start+8,bars*4)):
                actions={r:[n for n in proposals[r] if n[1]//480==beat] for r in oe.ROLES}
                ledger=nt.commit(ledger,actions,beat,brief)
                for role,notes in actions.items():decisions.extend(dict(role=role,beat=beat,note=n,origin='pretrained_only') for n in notes)
            trace.append(dict(start_beat=start,committed=sum(proposals.values(),[])))
    assert sorted(ledger)==sorted(d['note'] for d in decisions)
    return dict(notes=ledger,decisions=decisions,trace=trace,brief=nt.asdict(brief),seed=seed,dataset_prefix=[])


def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.manual_seed(SEED)
    key=json.loads((SOURCE/'panel_key.json').read_text())
    assert key['labels']['B']['variant']=='phrase'
    model=nt.load_model()
    assert not any('lora_' in name for name,_ in model.named_parameters())
    assert not any(p.requires_grad for p in model.parameters())
    initial_hash=fingerprint(model)
    expected=torch.load(ROOT/'outputs/supported_ensemble_run_v1/participation.pt',map_location='cpu',weights_only=False)['base_sha256']
    if adapter_compatible_base_hash(model)!=expected:raise ValueError('Original pretrained backbone identity mismatch')
    agents={r:oe.Musician().eval() for r in oe.ROLES}
    assert all(torch.count_nonzero(a.head.weight)==0 and torch.count_nonzero(a.head.bias)==0 for a in agents.values())
    ep=compose(model,agents);write_json(OUT/'base_episode.json',ep)
    normal=compose(model,agents,2,SEED+99);reverse=compose(model,agents,2,SEED+99,order=list(reversed(oe.ROLES)))
    assert normal['notes']==reverse['notes']
    if fingerprint(model)!=initial_hash:raise ValueError('Base weights changed')
    print('Untouched pretrained notes generated:',{r:sum(n[3]==i for n in ep['notes']) for i,r in enumerate(oe.ROLES)},flush=True)
    nt.midi_file(OUT/'base.mid',ep['notes'],bpm=100);render(OUT/'base.mid',OUT/'base_raw.wav')
    subprocess.run(['ffmpeg','-y','-v','error','-i',str(OUT/'base_raw.wav'),'-af','apad,atrim=duration=21.6,afade=t=out:st=20.6:d=1','-c:a','pcm_s24le',str(OUT/'base_trim.wav')],check=True)
    sources={'base':OUT/'base_trim.wav','clip_b':SOURCE/'B_mix.wav'}
    levels={k:loudness(path) for k,path in sources.items()}
    target=min([-18]+[-2-(peak-integrated) for integrated,peak in levels.values()])
    checks={}
    for name,path in sources.items():
        gain=target-levels[name][0];wav=OUT/f'{name}.wav';mp3=OUT/f'{name}.mp3'
        subprocess.run(['ffmpeg','-y','-v','error','-i',str(path),'-af',f'volume={gain:.9f}dB','-c:a','pcm_s24le',str(wav)],check=True)
        subprocess.run(['ffmpeg','-y','-v','error','-i',str(wav),'-q:a','2',str(mp3)],check=True)
        decoded=OUT/f'{name}_decoded.wav';subprocess.run(['ffmpeg','-y','-v','error','-i',str(mp3),str(decoded)],check=True)
        pcm,sr=sf.read(decoded);decoded.unlink()
        assert np.isfinite(pcm).all() and np.max(np.abs(pcm))<1
        checks[name]=dict(seconds=len(pcm)/sr,mp3_decoded_peak=float(np.max(np.abs(pcm))),measured_lufs=loudness(wav)[0],constant_gain_db=gain,sha256=hashlib.sha256(mp3.read_bytes()).hexdigest())
    shutil.copy2(SOURCE/'B_mix.mid',OUT/'clip_b.mid')
    write_json(OUT/'verification.json',dict(seed=SEED,bars=8,bpm=100,pretrained_model='SkyTNT tv2o-medium',
        base_sha256=initial_hash,adapter_compatible_base_sha256=expected,no_lora_modules=True,no_project_trained_weights_loaded_into_model=True,
        no_planner_or_conditioning=True,neutral_residual_head=True,backbone_unchanged=True,
        agent_order_independent=True,midi_action_agreement=True,base_notes=len(ep['notes']),
        clip_b_source=str(SOURCE/'B_mix.wav'),shared_target_lufs=target,audio=checks,
        limitation='This isolates our project training under the existing trio sampler, instrument ranges and event limits; it is not unrestricted upstream model generation.'))
    (OUT/'listen.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,"><title>Original pretrained model vs Clip B</title><style>body{font:18px system-ui;background:#101724;color:#edf2fa;max-width:850px;margin:35px auto;padding:20px;line-height:1.6}section{background:#1d2b40;border-radius:14px;padding:22px;margin:24px 0}audio{width:100%}button{font:inherit;padding:10px;margin-top:15px}a{color:#9dd8ff}</style>
<h1>Original pretrained model vs Clip B</h1><p>Same seed, eight bars, 100 BPM, organ / strings / cello, instrument limits and sound library. Both mixes have matched loudness. Every note is generated.</p><section><h2>Original pretrained model</h2><p>No project fine-tuning, instrument adapters, learned planner or partner conditioning. The model still has its original pretraining.</p><audio controls preload="metadata" src="base.mp3"></audio><button>Play from beginning</button></section><section><h2>Clip B — phrase fine-tuning</h2><p>The exact notes from the Clip B you preferred, with matched playback loudness.</p><audio controls preload="metadata" src="clip_b.mp3"></audio><button>Play from beginning</button></section><p><a href="verification.json">Generation and audio checks</a> · <a href="base.mid">Base MIDI</a> · <a href="clip_b.mid">Clip B MIDI</a></p><script>for(const s of document.querySelectorAll('section')){const a=s.querySelector('audio');s.querySelector('button').onclick=()=>{a.currentTime=0;a.play().catch(()=>{})};a.addEventListener('play',()=>{for(const other of document.querySelectorAll('audio'))if(other!==a)other.pause()})}</script></html>''',encoding='utf-8')
    print('Base versus Clip B ready:',OUT/'listen.html',flush=True)


if __name__=='__main__':main()
