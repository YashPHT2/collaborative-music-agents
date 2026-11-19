"""Predetermined-seed initial/selected/final rock comparison; no sample selection."""
import os
os.environ.setdefault('USE_TF','0');os.environ.setdefault('USE_FLAX','0')
import copy,gzip,hashlib,json,subprocess
from pathlib import Path
import mido,numpy as np,soundfile as sf,torch
import rock_ensemble as rock
import orchestra_learning as l
import offline_ensemble as oe
import note_trio as nt
from generate_full_orchestra import roundtrip
from generate_piano import render
from audit_projected_roles import loudness
from train_ensemble_repair import write_json

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs/rock_listening_v1'
SEED=9976101;BARS=12;BPM=120;RATE=44100;SECONDS=BARS*4*60/BPM+2
SOURCES={'A':'selected.pt','B':'initial.pt','C':'final128.pt'}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def commit(ledger,actions,beat):
    if set(actions)!=set(rock.ROLES):raise ValueError('All roles must commit synchronously')
    added=[]
    for role,s in rock.ROLES.items():
        notes=actions[role]
        assert len(notes)<=s['cap']
        own=[n for n in ledger if n[3]==s['id']]+notes
        for n in notes:
            assert n[3]==s['id'] and beat*480<=n[1]<(beat+1)*480 and n[1]%30==0
            assert 60<=n[2]<=3840 and n[2]%60==0 and n[1]+n[2]<=BARS*1920
            assert n[4] in s['pitches'] and n[5] in range(32,125,4)
            if role=='drums':assert n[2]==60
            active=[p for p in own if p[1]<=n[1]<p[1]+p[2]]
            assert len(active)<=s['polyphony'] and len({p[4] for p in active})==len(active)
        added.extend(notes)
    return sorted(ledger+added,key=lambda n:(n[1],n[3],n[4]))

@torch.no_grad()
def compose(model,adapters,planner,conditions,actors,goal,variant):
    ledger=[];decisions=[];trace=[];plans=[];previous=torch.zeros(l.SIZE,device=model.device);hidden=None
    brief=nt.Brief(BARS,BPM,0,False)
    for start in range(0,BARS*4,8):
        summary=l.description(l.visible(ledger,start),start//4-1,BARS,list(rock.ROLES)).to(model.device) if start else torch.zeros_like(previous)
        for bar in range(start//4,start//4+2):
            inputs=torch.cat([previous,summary,goal,previous.new_tensor([bar/(BARS-1)])])
            pred,hidden=planner(inputs[None,None],hidden);previous=pred[0,0];plans.append(previous)
        for beat in range(start,start+8):
            past=l.visible(ledger,beat);actions={}
            for role,s in rock.ROLES.items():
                adapters.activate(s['family'])
                delta=conditions(plans[beat//4],torch.zeros(1,8,device=model.device),role,goal)
                sampler=rock.Sampler(oe.ConditionedModel(model,delta),role,SEED+beat*1009+s['id']*100003,brief,beat,past)
                made,_=sampler.propose(actors[role],torch.zeros(32))
                for n in made:n[3]=s['id']
                actions[role]=made;decisions.extend(dict(role=role,beat=beat,note=n) for n in made)
            ledger=commit(ledger,actions,beat)
        print(variant,start//4+2,'/',BARS,'bars',len(ledger),'notes',flush=True)
        trace.append(dict(start=start,notes=len(ledger)))
    assert sorted(ledger)==sorted(d['note'] for d in decisions)
    return dict(notes=ledger,decisions=decisions,trace=trace,seed=SEED,bars=BARS,bpm=BPM)

def export(directory,episode):
    midi=mido.MidiFile(ticks_per_beat=480);meta=mido.MidiTrack();midi.tracks.append(meta)
    meta.append(mido.MetaMessage('set_tempo',tempo=round(60000000/BPM)));meta.append(mido.MetaMessage('time_signature',numerator=4,denominator=4))
    for role,s in rock.ROLES.items():
        notes=[n for n in episode['notes'] if n[3]==s['id']]
        track=mido.MidiTrack();track.append(mido.MetaMessage('track_name',name=role))
        track.append(mido.Message('program_change',channel=s['channel'],program=s['patch']))
        events=[]
        for n in notes:
            events.extend([(n[1],1,mido.Message('note_on',channel=s['channel'],note=n[4],velocity=n[5])),(n[1]+n[2],0,mido.Message('note_off',channel=s['channel'],note=n[4],velocity=0))])
        previous=0
        for tick,_,msg in sorted(events,key=lambda row:(row[0],row[1])):
            msg.time=tick-previous;track.append(msg);previous=tick
        midi.tracks.append(track)
        single=mido.MidiFile(ticks_per_beat=480);single.tracks.append(copy.deepcopy(meta));single.tracks.append(copy.deepcopy(track));single.save(directory/(role+'.mid'))
    path=directory/'band.mid';midi.save(path)
    for (role,s),track in zip(rock.ROLES.items(),mido.MidiFile(path).tracks[1:]):
        roundtrip(track,[n for n in episode['notes'] if n[3]==s['id']],s['id'])
        assert all(msg.channel==s['channel'] for msg in track if msg.type in ['note_on','note_off'])
    return path

def page(ready):
    cards=[]
    for v in SOURCES:
        cards.append(f'<section><h2>Clip {v}</h2><audio controls preload="metadata" src="{v}.mp3"></audio><p><a download href="{v}.mp3">MP3</a> · <a download href="{v}/band.mid">MIDI</a></p></section>' if v in ready else f'<section><h2>Clip {v}</h2><p>Preparing—refresh later.</p></section>')
    js='''for(const a of document.querySelectorAll('audio'))a.onplay=()=>{for(const b of document.querySelectorAll('audio'))if(a!==b)b.pause()};const q=document.getElementById('choice'),key='rock-checkpoint-comparison-v1';q.value=localStorage.getItem(key)||'';q.onchange=()=>localStorage.setItem(key,q.value);document.getElementById('save').onclick=()=>{if(!q.value)return;const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({schema:key,seed:9976101,choice:q.value},null,2)],{type:'application/json'}));a.download='rock-checkpoint-comparison-v1.json';a.click()};'''
    (OUT/'player_check.js').write_text(js)
    (OUT/'listen.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Rock checkpoint comparison</title><style>body{font:18px/1.6 system-ui;max-width:850px;margin:30px auto;padding:20px;background:#14212b;color:#edf5ef}section{background:#233746;padding:24px;margin:22px 0;border-radius:16px}audio{width:100%}select,button{font:inherit;padding:10px}a{color:#a6e7cb}</style><h1>Rock ensemble comparison</h1><p>Three checkpoints, one fixed seed. Identical 120 BPM, instruments and rendering settings, with matched loudness. All notes are generated by the role policies. This is a rock pilot; happy mood has not been trained.</p>'''+''.join(cards)+'''<label>Which would you rather listen to? <select id="choice"><option value="">Choose</option><option>A</option><option>B</option><option>C</option><option>Same</option><option>Neither</option></select></label><p><button id="save">Download choice</button></p><script>'''+js+'</script></html>',encoding='utf-8')

def main():
    OUT.mkdir(exist_ok=True);rock.configure();torch.set_num_threads(4);torch.manual_seed(SEED)
    checkpoint_root=ROOT/'outputs/rock_training_v1/checkpoints'
    checkpoints={v:torch.load(checkpoint_root/name,map_location='cpu',weights_only=False) for v,name in SOURCES.items()}
    assert [checkpoints[v]['state']['step'] for v in SOURCES]==[32,0,128]
    assert all(p['roles']==rock.ROLES and p['format_version']=='rock-pilot-v1' for p in checkpoints.values())
    with gzip.open(ROOT/'outputs/rock_training_v1/rock_examples.json.gz','rt') as f:windows=json.load(f)
    goal=np.mean([l.style([n for n in w['notes'] if n[3]!=3],w['available_roles']).numpy() for w in windows if w['split']=='train'],axis=0)
    protocol=dict(seed=SEED,bars=BARS,bpm=BPM,goal_from_training_only=goal.tolist(),all_samples_retained=True,new_training=False,checkpoint_sha256={v:sha(checkpoint_root/name) for v,name in SOURCES.items()},soundfont_sha256=sha(ROOT/'assets/soundfont.sf2'))
    path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    write_json(path,protocol);write_json(OUT/'organizer_mapping.json',{v:dict(source=name,step=checkpoints[v]['state']['step']) for v,name in SOURCES.items()});page({})
    cached=all((OUT/v/'episode.json').exists() for v in SOURCES) and (OUT/'notes_verified.json').exists()
    if cached:
        verified=json.loads((OUT/'notes_verified.json').read_text());assert verified['frozen_backbone_unchanged'] and verified['seed']==SEED
        for v in SOURCES:assert verified['clips'][v]['source_sha256']==protocol['checkpoint_sha256'][v]
    else:
        model=nt.load_model();adapters=l.Adapters(model);before=adapters.base_hash();planner=l.Planner().to(model.device);conditions=l.Conditions().to(model.device)
        assert all(p['base_sha256']==before for p in checkpoints.values())
        actors={r:oe.Musician().eval() for r in rock.ROLES}
    raws={};episodes={}
    for v,p in checkpoints.items():
        directory=OUT/v;directory.mkdir(exist_ok=True);epfile=directory/'episode.json'
        if not cached:
            adapters.load(p['adapters']);planner.load_state_dict(p['planner']);conditions.load_state_dict(p['conditions']);planner.eval();conditions.eval()
        episode=json.loads(epfile.read_text()) if epfile.exists() else compose(model,adapters,planner,conditions,actors,torch.tensor(goal,device=model.device),v)
        write_json(epfile,episode);episodes[v]=episode;path=export(directory,episode)
        render(path,directory/'raw.wav');audio,sr=sf.read(directory/'raw.wav',always_2d=True)
        assert sr==RATE;fixed=np.zeros((round(SECONDS*RATE),2));n=min(len(audio),len(fixed));fixed[:n]=audio[:n]
        fixed[-RATE:]*=np.linspace(1,0,RATE)[:,None];raw=directory/'fixed.wav';sf.write(raw,fixed,RATE,subtype='FLOAT');raws[v]=raw
    levels={v:loudness(p) for v,p in raws.items()};target=min([-22.]+[-2-(peak-lufs) for lufs,peak in levels.values() if np.isfinite(lufs)])
    checks={}
    for v,raw in raws.items():
        gain=target-levels[v][0] if np.isfinite(levels[v][0]) else 0.;wav=OUT/(v+'.wav');mp3=OUT/(v+'.mp3')
        subprocess.run(['ffmpeg','-y','-v','error','-i',str(raw),'-af',f'volume={gain:.9f}dB','-c:a','pcm_s24le',str(wav)],check=True)
        subprocess.run(['ffmpeg','-y','-v','error','-i',str(wav),'-q:a','2',str(mp3)],check=True)
        decoded=OUT/(v+'_decoded.wav');subprocess.run(['ffmpeg','-y','-v','error','-i',str(mp3),str(decoded)],check=True)
        audio,sr=sf.read(decoded);decoded.unlink();assert np.isfinite(audio).all() and np.max(np.abs(audio))<1
        checks[v]=dict(sha256=sha(mp3),seconds=len(audio)/sr,lufs=loudness(wav)[0],peak=float(np.max(np.abs(audio))),notes=len(episodes[v]['notes']),per_role={r:sum(n[3]==s['id'] for n in episodes[v]['notes']) for r,s in rock.ROLES.items()},midi_action_roundtrip=True)
    if not cached:assert adapters.base_hash()==before
    write_json(OUT/'generation.json',dict(status='complete',clips=checks,frozen_backbone_unchanged=True));page(checks)
    print('COMPARISON READY',flush=True)

if __name__=='__main__':main()
