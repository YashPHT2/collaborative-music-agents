"""New step-224 sketches plus repeated anchors for listener calibration."""
import hashlib,json,shutil,sys
from pathlib import Path
import generate_checkpoint_music as generate

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/taste_collection_v1'
SEEDS=[9975301,9975302,9975303]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=True)
    generated=OUT/'new_sketches'
    if not (generated/'generation.json').exists():
        sys.argv=['generate_checkpoint_music.py','--checkpoint',str(ROOT/'outputs/preferred_model_v1/step224.pt'),'--output',str(generated),'--seeds',*map(str,SEEDS)]
        generate.main()
    info=json.loads((generated/'generation.json').read_text())
    if info['status']!='complete':raise ValueError('Generation incomplete; preserve files and finish generation before publishing')
    audit=ROOT/'outputs/orchestra_reward_audit_v3'
    prior=json.loads((audit/'protocol.json').read_text())
    # Fixed shuffled order. Anchors are repeat ratings, never new independent evidence.
    order=[('new',1),('anchor',0),('anchor',2),('new',2),('anchor',1),('new',3)]
    mapping={};cards=[]
    for i,(kind,n) in enumerate(order,1):
        key=f'clip_{i:02d}'
        if kind=='new':
            source=generated/f'piece_{n:02d}.mp3';seed=SEEDS[n-1];model='step224';repeat=False
        else:
            source=audit/f'clip_{n}.mp3';seed=9919101;model=['laptop','step128','step224'][n];repeat=True
            assert sha(source)==prior['source_hashes'][['laptop','a100_128','a100_224'][n]]
        shutil.copy2(source,OUT/(key+'.mp3'))
        mapping[key]=dict(model=model,seed=seed,sha256=sha(source),repeat_anchor=repeat,group=f'seed:{seed}',split='preference_development')
        controls=''.join(f'<label>{label} <select data-category="{category}"><option value="">Unrated</option>'+''.join(f'<option>{v}</option>' for v in range(1,6))+'</select></label>' for category,label in [('enjoyment','Enjoyment'),('fit','Parts fitting together'),('development','Musical development'),('warmth','Warmth / positive mood')])
        cards.append(f'<section data-id="{key}"><h2>Clip {i}</h2><audio controls preload="metadata" src="{key}.mp3"></audio>{controls}</section>')
    (OUT/'organizer_mapping.json').write_text(json.dumps(mapping,indent=2))
    (OUT/'protocol.json').write_text(json.dumps(dict(schema='taste-collection-v1',new_seeds=SEEDS,checkpoint_sha256=info['checkpoint_sha256'],new_clips=3,repeat_anchors=3,split='preference_development',heldout_ballots_reused=False,reward_validated=False,rl_training_allowed=False),indent=2))
    script='''const key='taste-collection-v1';let saved=JSON.parse(localStorage.getItem(key)||'{}');for(const s of document.querySelectorAll('section'))for(const q of s.querySelectorAll('select')){q.value=saved[s.dataset.id]?.[q.dataset.category]||'';q.onchange=()=>{(saved[s.dataset.id]??={})[q.dataset.category]=q.value;localStorage.setItem(key,JSON.stringify(saved))}}for(const a of document.querySelectorAll('audio'))a.onplay=()=>{for(const b of document.querySelectorAll('audio'))if(a!==b)b.pause()};document.getElementById('save').onclick=()=>{const rows=[];for(const s of document.querySelectorAll('section')){const ratings={};for(const q of s.querySelectorAll('select')){if(!q.value){document.getElementById('status').textContent='Please rate all four questions for every clip.';return}ratings[q.dataset.category]=Number(q.value)}rows.push({id:s.dataset.id,ratings})}const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({schema:'taste-collection-v1',rows},null,2)],{type:'application/json'}));a.download='taste-collection-v1.json';a.click()};'''
    (OUT/'player_check.js').write_text(script)
    (OUT/'listen.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Music taste ratings</title><style>body{font:18px/1.6 system-ui;background:#14212b;color:#ecf4ee;max-width:850px;margin:30px auto;padding:20px}section{padding:24px;background:#233746;border-radius:16px;margin:24px 0}audio{width:100%}label{display:block;margin:12px 0}select,button{font:inherit;padding:8px}</style><h1>What music do you enjoy?</h1><p>Six clips: new sketches and some earlier examples for consistency. Rate each from 1 (poor / very little) to 5 (excellent / very much). Judge the sound you hear. Similar parts can fit together even if you dislike the mood.</p>'''+''.join(cards)+'<button id="save">Download ratings</button><p id="status" role="status"></p><script>'+script+'</script></html>',encoding='utf-8')
    print('TASTE COLLECTION READY',flush=True)

if __name__=='__main__':main()
