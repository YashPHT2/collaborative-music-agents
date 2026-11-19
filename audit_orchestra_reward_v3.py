"""Frozen reward audit on actual 19-role generated audio; never trains a policy."""
import argparse, hashlib, json, shutil
from pathlib import Path
import numpy as np
import soundfile as sf
from preference_reward import PreferenceReward

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/orchestra_reward_audit_v3'
LISTEN=ROOT/'outputs/a100_phrase_snapshot_v1/listening'
SOURCES={'laptop':LISTEN/'initial.mp3','a100_128':LISTEN/'trained.mp3',
         'a100_224':LISTEN/'latest_224/latest.mp3',
         **{f'new_seed_{i}':LISTEN/f'more/piece_0{i}.mp3' for i in range(1,4)}}

def write(path,data):path.write_text(json.dumps(data,indent=2),encoding='utf-8')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(version=3,seed=145301,source_hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in SOURCES.items()},
        synthetic_tests=['short_loop','block_shuffle','silence','noise'],gain_invariance_tolerance=0.02,
        structural_min_win_rate=.8,all_sanity_controls_must_pass=True,
        human_evidence='One explicit matched-seed preference: a100_128 > laptop; insufficient for validation',
        threshold_tuning_on_this_audit=False,policy_training=False)
    if (OUT/'protocol.json').exists() and json.loads((OUT/'protocol.json').read_text())!=protocol:raise ValueError('Audit inputs changed; use new version')
    write(OUT/'protocol.json',protocol)
    model=PreferenceReward();rows=[]
    rng=np.random.default_rng(protocol['seed'])
    for name,path in SOURCES.items():
        y,sr=sf.read(path);block=2*sr;chunks=[y[i:i+block] for i in range(0,len(y)//block*block,block)]
        shuffled=np.concatenate([chunks[i] for i in rng.permutation(len(chunks))]+[y[len(chunks)*block:]])
        loop=np.tile(y[:block],(int(np.ceil(len(y)/block)),1))[:len(y)] if y.ndim==2 else np.tile(y[:block],int(np.ceil(len(y)/block)))[:len(y)]
        variants=dict(original=y,short_loop=loop,block_shuffle=shuffled,silence=np.zeros_like(y),
            noise=rng.normal(0,.03,y.shape),quieter=y*.5)
        scores={}
        for kind,audio in variants.items():
            cache=OUT/f'{name}_{kind}.json'
            if cache.exists():result=json.loads(cache.read_text())
            else:
                result=model.evaluate(audio,sr);write(cache,result)
            scores[kind]=result['preference'];print(name,kind,scores[kind],flush=True)
        rows.append(dict(id=name,scores=scores,structural_wins={k:scores['original']>scores[k] for k in ['short_loop','block_shuffle']},
            sanity_pass=scores['original']>max(scores['silence'],scores['noise']),
            gain_invariant=abs(scores['original']-scores['quieter'])<=protocol['gain_invariance_tolerance']))
        write(OUT/'results.json',rows)
    wins=[v for row in rows for v in row['structural_wins'].values()]
    structural=float(np.mean(wins));sanity=all(r['sanity_pass'] for r in rows);gain=all(r['gain_invariant'] for r in rows)
    agreement=rows[1]['scores']['original']>rows[0]['scores']['original']
    write(OUT/'acceptance.json',dict(automated_gate_passed=structural>=.8 and sanity and gain,
        structural_win_rate=structural,sanity_pass=sanity,gain_invariance_pass=gain,
        known_single_preference_agreement=agreement,human_preference_gate_passed=False,generated_domain_gate_passed=False,
        rl_training_allowed=False,reason='Only one matched human preference; no independent preference test set or 19-role symbolic reward artifact.'))
    # Individually rate original compositions; corruptions stay diagnostic only.
    cards=[]
    for i,(name,path) in enumerate(SOURCES.items()):
        shutil.copy2(path,OUT/f'clip_{i}.mp3')
        cards.append(f'<section data-id="{name}"><h2>Clip {i+1}</h2><audio controls src="clip_{i}.mp3"></audio>'+''.join(f'<label>{label} <select data-key="{key}"><option value="">Rate</option>'+''.join(f'<option>{j}</option>' for j in range(1,6))+'</select></label>' for key,label in [('enjoyment','Enjoyment'),('fit','Parts fitting together'),('development','Musical development'),('warmth','Warmth')])+'</section>')
    (OUT/'listen.html').write_text('<!doctype html><meta charset="utf-8"><title>Original orchestra ratings</title><style>body{font:18px system-ui;max-width:850px;margin:40px auto;background:#14212b;color:white}section{padding:20px;background:#233746;margin:20px}audio{width:100%}label{display:inline-block;margin:12px}button,select{font:inherit}</style><h1>Rate the original music</h1><p>1 = poor, 5 = excellent. These are existing original agent compositions, not newly trained outputs. Rate only clips you listen to.</p>'+''.join(cards)+'''<button onclick="save()">Download ratings</button><script>for(const a of document.querySelectorAll('audio'))a.onplay=()=>{for(const b of document.querySelectorAll('audio'))if(a!==b)b.pause()};function save(){let rows=[...document.querySelectorAll('section')].map(s=>({id:s.dataset.id,ratings:Object.fromEntries([...s.querySelectorAll('select')].filter(x=>x.value).map(x=>[x.dataset.key,+x.value]))})).filter(r=>Object.keys(r.ratings).length);const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({schema:'orchestra-ratings-v3',rows},null,2)],{type:'application/json'}));a.download='orchestra-ratings-v3.json';a.click()}</script>''',encoding='utf-8')
    print(json.dumps(json.loads((OUT/'acceptance.json').read_text()),indent=2))

if __name__=='__main__':main()
