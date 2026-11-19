"""Render baseline and equal-update final candidates, with failures disclosed."""
import json,subprocess
from pathlib import Path
import numpy as np,soundfile as sf
import generate_rock_comparison as g

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs/rock_history_listening_v1';RATE=44100
ORDERS=[['control_final','baseline','mixed_final'],['mixed_final','control_final','baseline'],['baseline','mixed_final','control_final']]

def page(protocol,results):
    cards=[]
    for i,seed in enumerate(protocol['seeds'],1):
        key=f'seed_{i:02d}'
        players=''.join(f'<article><h3>Clip {v}</h3><audio controls preload="metadata" src="{key}/{v}.mp3?v={results[key][v]["sha256"][:12]}"></audio><p><a download href="{key}/{v}.mp3">MP3</a> · <a download href="{key}/{v}/band.mid">MIDI</a></p></article>' for v in ['A','B','C'])
        cards.append(f'<section data-id="{key}"><h2>Comparison {i}</h2>'+players+'<label>Which would you rather listen to? <select><option value="">Choose</option><option>A</option><option>B</option><option>C</option><option>Same</option><option>Neither</option></select></label></section>')
    script='''const key='rock-history-listening-v1';let stored=JSON.parse(localStorage.getItem(key)||'{}');for(const s of document.querySelectorAll('section[data-id]')){const q=s.querySelector('select');q.value=stored[s.dataset.id]||'';q.onchange=()=>{stored[s.dataset.id]=q.value;localStorage.setItem(key,JSON.stringify(stored))}}for(const a of document.querySelectorAll('audio'))a.onplay=()=>{for(const b of document.querySelectorAll('audio'))if(a!==b)b.pause()};document.getElementById('save').onclick=()=>{const rows=[...document.querySelectorAll('section[data-id]')].map(s=>({comparison:s.dataset.id,choice:s.querySelector('select').value}));if(rows.some(r=>!r.choice)){document.getElementById('status').textContent='Please answer all three comparisons.';return}const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({schema:key,rows},null,2)],{type:'application/json'}));a.download='rock-history-listening-v1.json';a.click()};'''
    (OUT/'player_check.js').write_text(script)
    (OUT/'listen.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Rock history-training comparison</title><style>body{font:18px/1.6 system-ui;max-width:900px;margin:30px auto;padding:20px;background:#14212b;color:#edf5ef}section{background:#233746;padding:24px;border-radius:16px;margin:24px 0}article{border-top:1px solid #647581;padding:10px 0}audio{width:100%}button,select{font:inherit;padding:10px}a{color:#a6e7cb}</style><h1>Did the new training help?</h1><p>Each comparison uses one seed with three versions: original rock step 128, real-history control training, and mixed-history training. The candidates received the same 64 additional updates. Clip identities are shuffled between comparisons.</p><p>All notes are AI generated from the beginning, with no reference prompt. Each clip contains six bars at 120 BPM plus a release tail, about 14 seconds total. Instruments and loudness are matched within each comparison.</p><p><strong>Experimental results:</strong> neither trained candidate passed the validation guard, so the original checkpoint remains selected. These final candidates are included to diagnose their sound, not presented as established improvements.</p>'''+''.join(cards)+'''<p><button id="save">Download choices</button></p><p id="status" role="status"></p><script>'''+script+'</script></html>',encoding='utf-8')

def main():
    g.rock.configure();manifest=json.loads((OUT/'remote_manifest.json').read_text());assert manifest['frozen_backbone_unchanged'] and manifest['no_new_training']
    g.BARS=6;g.BPM=120;seconds=14;results={};mapping={}
    protocol=dict(schema='rock-history-listening-v1',seeds=manifest['seeds'],bars=6,bpm=120,reference_prefix=False,models=manifest['models'],all_samples_retained=True,
        candidate_status='Final experimental candidates; validation selection failed. Original baseline preserved.',soundfont_sha256=g.sha(ROOT/'assets/soundfont.sf2'))
    path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    g.write_json(path,protocol)
    for i,(seed,order) in enumerate(zip(manifest['seeds'],ORDERS),1):
        key=f'seed_{i:02d}';folder=OUT/key;folder.mkdir(exist_ok=True);raws={};episodes={};mapping[key]=dict(seed=seed)
        for v,model_name in zip(['A','B','C'],order):
            name=f'{model_name}_{seed}';info=manifest['artifacts'][name];source=OUT/'episodes'/(name+'.json')
            assert g.sha(source)==info['sha256'];ep=json.loads(source.read_text());assert ep['seed']==seed and not ep['reference_prefix']
            assert sorted(ep['notes'])==sorted(row['note'] for row in ep['decisions'])
            episodes[v]=ep;mapping[key][v]=model_name;directory=folder/v;directory.mkdir(exist_ok=True)
            midi=g.export(directory,ep);raw=directory/'raw.wav'
            if not raw.exists():g.render(midi,raw)
            audio,sr=sf.read(raw,always_2d=True);assert sr==RATE
            fixed=np.zeros((seconds*RATE,2));n=min(len(audio),len(fixed));fixed[:n]=audio[:n];fixed[-RATE:]*=np.linspace(1,0,RATE)[:,None]
            path=directory/'fixed.wav';sf.write(path,fixed,RATE,subtype='FLOAT');raws[v]=path
        levels={v:g.loudness(path) for v,path in raws.items()};target=min([-22.]+[-2-(peak-lufs) for lufs,peak in levels.values() if np.isfinite(lufs)])
        results[key]={}
        for v,raw in raws.items():
            gain=target-levels[v][0] if np.isfinite(levels[v][0]) else 0.;wav=folder/(v+'.wav');mp3=folder/(v+'.mp3')
            subprocess.run(['ffmpeg','-y','-v','error','-i',str(raw),'-af',f'volume={gain:.9f}dB','-c:a','pcm_s24le',str(wav)],check=True)
            # Re-measure after BS.1770 gating; a single gain estimate can differ
            # slightly once quiet blocks enter/leave the absolute gate.
            for correction_pass in range(2):
                actual,peak=g.loudness(wav)
                if not np.isfinite(actual) or abs(actual-target)<=.03:break
                correction=min(target-actual,-1.5-peak)
                corrected=folder/(v+'_corrected.wav')
                subprocess.run(['ffmpeg','-y','-v','error','-i',str(wav),'-af',f'volume={correction:.9f}dB','-c:a','pcm_s24le',str(corrected)],check=True)
                corrected.replace(wav)
            subprocess.run(['ffmpeg','-y','-v','error','-i',str(wav),'-q:a','2',str(mp3)],check=True)
            decoded=folder/(v+'_decoded.wav');subprocess.run(['ffmpeg','-y','-v','error','-i',str(mp3),str(decoded)],check=True)
            audio,sr=sf.read(decoded);decoded.unlink();assert np.isfinite(audio).all() and np.max(np.abs(audio))<1
            lufs=g.loudness(wav)[0]
            results[key][v]=dict(sha256=g.sha(mp3),seconds=len(audio)/sr,lufs=float(lufs) if np.isfinite(lufs) else None,silent=not np.any(audio),notes=len(episodes[v]['notes']),midi_action_roundtrip=True,decoded_peak=float(np.max(np.abs(audio))))
        print('READY',key,flush=True)
    g.write_json(OUT/'organizer_mapping.json',mapping);g.write_json(OUT/'generation.json',dict(status='complete',comparisons=results,new_training=False));page(protocol,results)
    print('HISTORY COMPARISON READY',flush=True)

if __name__=='__main__':main()
