"""Three predeclared unseen seed groups; matched, blinded checkpoint comparisons."""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import orchestra_learning as l
import note_trio as nt
import offline_ensemble as oe
import evaluate_trained_orchestra as preview
from audit_projected_roles import loudness
from train_ensemble_repair import write_json

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/preference_holdout_v1'
SEEDS=[9975101,9975102,9975103]
SOURCES=[ROOT/'outputs/a100_phrase_snapshot_v1/checkpoints/phrases.pt',ROOT/'outputs/a100_progress_step224_v1/progress.pt']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def page(ready):
    cards=[]
    for i in range(3):
        name=f'pair_{i+1:02d}'
        if name not in ready:
            cards.append(f'<section><h2>Pair {i+1}</h2><p>Preparing… refresh later.</p></section>');continue
        cards.append(f'<section data-id="{name}"><h2>Pair {i+1}</h2>'+''.join(f'<p>Clip {v}</p><audio controls preload="metadata" src="{name}/{v}.mp3"></audio>' for v in ['A','B'])+'<label>Which would you rather listen to? <select><option value="">Choose</option><option>A</option><option>B</option><option>Same</option><option>Neither</option></select></label></section>')
    (OUT/'listen.html').write_text('''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>New music comparisons</title><style>body{font:18px/1.6 system-ui;max-width:850px;margin:35px auto;padding:20px;background:#14212b;color:#ecf4ee}section{background:#233746;padding:24px;margin:24px 0;border-radius:16px}audio{width:100%}select,button{font:inherit;padding:10px;margin:14px 0}</style><h1>Three new music comparisons</h1><p>Each pair uses the same new seed, instruments, tempo and loudness target. A/B order varies. Choose the music you prefer; Same and Neither are valid.</p>'''+''.join(cards)+'''<button id="save">Download choices</button><p id="status"></p><script>
const key='preference-holdout-v1';let stored=JSON.parse(localStorage.getItem(key)||'{}');
for(const s of document.querySelectorAll('section[data-id]')){const q=s.querySelector('select');q.value=stored[s.dataset.id]||'';q.onchange=()=>{stored[s.dataset.id]=q.value;localStorage.setItem(key,JSON.stringify(stored))}}
for(const a of document.querySelectorAll('audio'))a.onplay=()=>{for(const b of document.querySelectorAll('audio'))if(a!==b)b.pause()};
document.getElementById('save').onclick=()=>{const rows=[...document.querySelectorAll('section[data-id]')].map(s=>({pair:s.dataset.id,choice:s.querySelector('select').value})).filter(r=>r.choice);if(!rows.length){document.getElementById('status').textContent='Choose at least one pair first.';return}const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({schema:'preference-holdout-v1',rows},null,2)],{type:'application/json'}));a.download='preference-holdout-v1.json';a.click()};</script>''',encoding='utf-8')

def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.manual_seed(9975100)
    protocol=dict(schema='preference-holdout-v1',seeds=SEEDS,source_sha256=[sha(p) for p in SOURCES],bars=8,bpm=70,
        split='heldout_listener_evaluation',new_training=False,all_seeds_retained=True,
        prior_rating_groups=[9919101,9919102,9919103,9919104],
        note='Exploratory checkpoint comparison; three seed groups cannot validate a general taste reward.')
    old=OUT/'protocol.json'
    if old.exists() and json.loads(old.read_text())!=protocol:raise ValueError('Protocol changed')
    assert not set(SEEDS)&set(protocol['prior_rating_groups']);write_json(old,protocol)
    mapping={f'pair_{i+1:02d}':dict(seed=seed,A=(i%2),B=1-(i%2)) for i,seed in enumerate(SEEDS)}
    write_json(OUT/'organizer_mapping.json',mapping)
    statefile=OUT/'generation.json';ready=json.loads(statefile.read_text()).get('pairs',{}) if statefile.exists() else {};page(ready)
    saved=[torch.load(p,map_location='cpu',weights_only=False) for p in SOURCES]
    for key in ['format_version','roles','families','base_sha256','calm_profile','composition_mode']:
        if saved[0][key]!=saved[1][key]:raise ValueError('Unmatched settings: '+key)
    model=nt.load_model();adapters=l.Adapters(model);base=adapters.base_hash()
    if base!=saved[0]['base_sha256']:raise ValueError('Wrong backbone')
    planner=l.Planner().to(model.device);conditions=l.Conditions().to(model.device)
    actors={r:oe.Musician().eval() for r in l.ROLES}
    for pair,m in mapping.items():
        if pair in ready:continue
        directory=OUT/pair;directory.mkdir(exist_ok=True);preview.OUT=directory;raws={};episodes={}
        for variant in ['A','B']:
            source=saved[m[variant]];adapters.load(source['adapters']);planner.load_state_dict(source['planner']);conditions.load_state_dict(source['conditions']);planner.eval();conditions.eval()
            path=directory/(variant+'_episode.json')
            if path.exists():ep=json.loads(path.read_text())
            else:
                print('Generating',pair,variant,flush=True)
                ep=l.compose(model,adapters,actors,planner,conditions,8,m['seed'],source['composition_mode'],saved[0]['calm_profile'],progress=lambda bars,notes:print(pair,variant,bars,'/8 bars',notes,'notes',flush=True))
                write_json(path,ep)
            episodes[variant]=ep
            raw,parts,fade=preview.render(variant,ep);del parts,fade;raws[variant]=raw
        levels={v:loudness(p) for v,p in raws.items()};target=min([-22.]+[-2-(peak-lufs) for lufs,peak in levels.values() if np.isfinite(lufs)])
        checks={}
        for variant,raw in raws.items():
            gain=target-levels[variant][0] if np.isfinite(levels[variant][0]) else 0.
            wav=directory/(variant+'.wav');mp3=directory/(variant+'.mp3')
            subprocess.run(['ffmpeg','-y','-v','error','-i',str(raw),'-af',f'volume={gain:.9f}dB','-c:a','pcm_s24le',str(wav)],check=True)
            subprocess.run(['ffmpeg','-y','-v','error','-i',str(wav),'-q:a','2',str(mp3)],check=True)
            decoded=directory/(variant+'_decode.wav');subprocess.run(['ffmpeg','-y','-v','error','-i',str(mp3),str(decoded)],check=True)
            audio,sr=sf.read(decoded);decoded.unlink()
            assert np.isfinite(audio).all() and np.max(np.abs(audio))<1
            ep=episodes[variant];assert sorted(ep['notes'])==sorted(d['note'] for d in ep['decisions'])
            checks[variant]=dict(sha256=sha(mp3),seconds=len(audio)/sr,peak=float(np.max(np.abs(audio))),lufs=loudness(wav)[0],
                participation=preview.participation(ep['notes']),notes_match_actions=True,midi_roundtrip_passed=True)
        ready[pair]=checks;write_json(statefile,dict(status='generating',pairs=ready));page(ready)
        print('READY',pair,flush=True)
    assert adapters.base_hash()==base
    write_json(statefile,dict(status='complete',pairs=ready,frozen_backbone_unchanged=True));page(ready)
    print('ALL PAIRS READY',flush=True)

if __name__=='__main__':main()
