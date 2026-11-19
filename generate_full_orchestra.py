"""Nineteen pretrained note agents with explicit arrangement controls and sampled sounds."""
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import subprocess
import os
import shutil

import mido
import numpy as np
from scipy.signal import fftconvolve,lfilter
import soundfile as sf
import torch

import note_trio as nt
import offline_ensemble as oe
from accompany import EventSampler
from band import BandSampler,fingerprint
from audit_projected_roles import loudness
from train_ensemble_repair import write_json

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/full_orchestra_v1'
SEED=9901101
BARS=12
BPM=80
RATE=44100
TEMPO=round(60_000_000/BPM)
SECONDS=BARS*4*TEMPO/1_000_000+3
SCALE={0,2,4,5,7,9,11}
# name, family, GM program, register, polyphony, first active beat, onset grid
SPECS=[('pipe_organ','organ',19,48,84,3,0,[0,8]),
       ('violins','strings',40,67,88,2,8,[0]),('violas','strings',41,55,76,2,0,[0]),
       ('cellos','strings',42,43,64,1,0,[0]),('double_basses','strings',43,31,52,1,0,[0]),
       ('piano_anchor','pianos',0,48,67,2,0,[0,8]),('piano_motion','pianos',0,60,79,1,0,[0,8]),
       ('piano_pulse','pianos',0,67,84,1,16,[0,8]),('piano_counterline','pianos',0,72,88,1,24,[0,8]),
       ('flutes','woodwinds',73,67,88,1,16,[0]),('oboes','woodwinds',68,60,81,1,24,[0]),
       ('clarinets','woodwinds',71,55,79,1,24,[0]),('bassoons','woodwinds',70,43,64,1,16,[0]),
       ('french_horns','brass',60,48,72,2,24,[0]),('trombones','brass',57,43,67,1,32,[0]),
       ('tuba','brass',58,31,52,1,32,[0]),('synth_pad','synth',89,48,79,2,16,[0]),
       ('choir','choir',52,55,81,2,32,[0])]
# One more violin section brings the requested expansive string texture to two desks.
SPECS.insert(2,('second_violins','strings',40,60,81,2,16,[0]))
ROLES={name:dict(id=i,family=family,patch=patch,pitches=[p for p in range(low,high+1) if p%12 in SCALE],
                 polyphony=poly,cap=2 if name=='pipe_organ' else 1,entry=entry,grid=grid,track=i+1)
       for i,(name,family,patch,low,high,poly,entry,grid) in enumerate(SPECS)}
PATCHES=list(dict.fromkeys(s['patch'] for s in ROLES.values()))
# Shared-patch parts share only their model channel; separate track and global role
# identities preserve four independent piano parts and the two violin sections.
CHANNELS={patch:i if i<9 else i+1 for i,patch in enumerate(PATCHES)}
FAMILIES=list(dict.fromkeys(s['family'] for s in ROLES.values()))


class OrchestraSampler(BandSampler):
    def __init__(self,model,role,seed,brief,beat,history,roles=None,channels=None):
        EventSampler.__init__(self,model,seed,brief.bpm,.9)
        self.role_specs=ROLES if roles is None else roles
        self.model_channels=CHANNELS if channels is None else channels
        self.role=role;self.settings={**self.role_specs[role],'channel':self.model_channels[self.role_specs[role]['patch']]}
        self.role_id=self.role_specs[role]['id'];self.brief=brief;self.beat=beat;self.exploration=.01
        self.rows=self.rows[:1]
        for event in [['time_signature',0,0,0,3,1],['key_signature',0,0,0,7,0],['set_tempo',0,0,0,brief.bpm]]:
            self.rows.append(self.tok.event2tokens(event))
        for settings in self.role_specs.values():
            self.rows.append(self.tok.event2tokens(['patch_change',0,0,settings['track'],self.model_channels[settings['patch']],settings['patch']]))
        self.observed=[]
        for n in sorted(history,key=lambda n:(n[1],n[3],n[4])):
            source=list(self.role_specs.values())[n[3]];local=copy.deepcopy(n);local[3]=self.model_channels[source['patch']]
            self.append_note(local,source['track']);self.observed.append(local+[n[3]])

    def own(self,n):return len(n)==6 or n[6]==self.role_id

    def onsets(self):
        return [s for s in self.settings['grid'] if self.beat*480+s*30>=self.last_tick and
                sum(self.own(n) and n[1]<=self.beat*480+s*30<n[1]+n[2] for n in self.observed)<self.settings['polyphony']]

    def legal_parameters(self,kind,decoded):
        if kind!='pitch':return super().legal_parameters(kind,decoded)
        observed=self.observed
        try:
            # Other piano agents are distinct instruments, despite sharing the
            # same model channel and General MIDI patch in the prompt.
            self.observed=[n for n in observed if self.own(n)]
            return super().legal_parameters(kind,decoded)
        finally:self.observed=observed

    def constrain_values(self,kind,values,decoded):
        family=self.settings['family'];phase=self.beat/max(1,self.brief.bars*4-1)
        if kind=='velocity':
            low,high={'organ':(36,76),'pianos':(32,68),'strings':(32,72),'woodwinds':(32,64),
                      'brass':(32,64),'synth':(28,52),'choir':(32,64)}[family]
            high=int((low+12+(high-low-12)*phase)//4*4)
            return list(range(low,high+1,4))
        if kind=='duration':
            maximum=min(64 if family=='pianos' else 128,
                        (self.brief.bars*1920-self.beat*480-decoded['time2']*30)//30)
            minimum=4 if family=='pianos' else 16 if family=='woodwinds' else 32
            return list(range(min(minimum,maximum),maximum+1,2))
        return values


def commit(ledger,actions,beat,bars):
    if set(actions)!=set(ROLES):raise ValueError('Every part must submit privately')
    additions=[]
    for role,settings in ROLES.items():
        notes=actions[role]
        if len(notes)>settings['cap']:raise ValueError('Role event cap')
        for n in notes:
            if not (n[0]=='note' and n[3]==settings['id'] and beat*480<=n[1]<(beat+1)*480
                    and n[1]%30==0 and n[2]>0 and n[2]%60==0 and n[2]<=3840
                    and n[1]+n[2]<=bars*1920 and n[4] in settings['pitches'] and 1<=n[5]<=127):raise ValueError('Illegal role note')
        own=[n for n in ledger+notes if n[3]==settings['id']]
        for n in notes:
            active=[p for p in own if p[1]<=n[1]<p[1]+p[2]]
            if len(active)>settings['polyphony'] or len({p[4] for p in active})!=len(active):raise ValueError('Role overlap')
        additions.extend(notes)
    return sorted(ledger+additions,key=lambda n:(n[1],n[3],n[4]))


@torch.no_grad()
def compose(model,agents,bars=BARS,seed=SEED,order=None):
    brief=nt.Brief(bars,BPM,0,False);ledger=[];decisions=[];trace=[]
    for beat in range(bars*4):
        history=[n for n in ledger if n[1]<beat*480 and (n[1]>=(beat-24)*480 or n[1]+n[2]>beat*480)]
        actions={r:[] for r in ROLES}
        for role in order or ROLES:
            if beat<ROLES[role]['entry']:continue
            sampler=OrchestraSampler(model,role,seed+beat*1009+ROLES[role]['id']*100003,brief,beat,history)
            notes,_=sampler.propose(agents[role],torch.zeros(32))
            for n in notes:n[3]=ROLES[role]['id']
            actions[role]=notes;decisions.extend(dict(role=role,beat=beat,note=n) for n in notes)
        ledger=commit(ledger,actions,beat,bars)
        trace.append(dict(beat=beat,history_sha256=hashlib.sha256(json.dumps(history).encode()).hexdigest(),emitted={r:len(ns) for r,ns in actions.items()}))
        if (beat+1)%8==0:print('Orchestra generated beat',beat+1,'/',bars*4,flush=True)
    assert sorted(ledger)==sorted(d['note'] for d in decisions)
    return dict(notes=ledger,decisions=decisions,trace=trace,brief=nt.asdict(brief),seed=seed)


def midi_track(settings,notes,channel=0):
    track=mido.MidiTrack();track.append(mido.MetaMessage('track_name',name=next(r for r,s in ROLES.items() if s==settings)))
    track.append(mido.Message('program_change',channel=channel,program=settings['patch']));events=[]
    for n in notes:
        events.extend([(n[1],1,mido.Message('note_on',channel=channel,note=n[4],velocity=n[5])),
                       (n[1]+n[2],0,mido.Message('note_off',channel=channel,note=n[4],velocity=0))])
    previous=0
    for tick,_,message in sorted(events,key=lambda x:(x[0],x[1])):message.time=tick-previous;track.append(message);previous=tick
    return track


def roundtrip(track,expected,global_id):
    tick=0;active={};actual=[]
    for msg in track:
        tick+=msg.time
        if msg.type=='note_on' and msg.velocity:
            if msg.note in active:raise ValueError('Repeated active pitch')
            active[msg.note]=(tick,msg.velocity)
        elif msg.type=='note_off' or msg.type=='note_on' and not msg.velocity:
            start,velocity=active.pop(msg.note);actual.append(['note',start,tick-start,global_id,msg.note,velocity])
    assert not active and sorted(actual)==sorted(expected)


def export_midis(episode):
    full=mido.MidiFile(ticks_per_beat=480);meta=mido.MidiTrack();full.tracks.append(meta)
    meta.append(mido.MetaMessage('set_tempo',tempo=TEMPO));meta.append(mido.MetaMessage('time_signature',numerator=4,denominator=4))
    channels=[c for c in range(16) if c!=9]
    for role,settings in ROLES.items():
        notes=[n for n in episode['notes'] if n[3]==settings['id']]
        single=mido.MidiFile(ticks_per_beat=480);single.tracks.append(copy.deepcopy(meta));single.tracks.append(midi_track(settings,notes));path=OUT/f'{role}.mid';single.save(path)
        roundtrip(mido.MidiFile(path).tracks[1],notes,settings['id'])
        track=midi_track(settings,notes,channels[settings['id']%15]);track.insert(0,mido.MetaMessage('midi_port',port=settings['id']//15));full.tracks.append(track)
    full.save(OUT/'orchestra_multiport.mid')
    for role,track in zip(ROLES,mido.MidiFile(OUT/'orchestra_multiport.mid').tracks[1:]):
        settings=ROLES[role];roundtrip(track,[n for n in episode['notes'] if n[3]==settings['id']],settings['id'])


def render_role(role):
    path=OUT/f'{role}_raw.wav'
    local=list((ROOT/'tools/fluidsynth').rglob('fluidsynth.exe'))
    fluid=os.environ.get('FLUIDSYNTH_BIN') or (str(local[0]) if os.name=='nt' and local else shutil.which('fluidsynth'))
    if not fluid:raise FileNotFoundError('Install native fluidsynth or set FLUIDSYNTH_BIN')
    subprocess.run([str(fluid),'-ni','-F',str(path),'-r',str(RATE),'-g','0.3',str(ROOT/'assets/soundfont.sf2'),str(OUT/f'{role}.mid')],check=True,capture_output=True)
    data,sr=sf.read(path,always_2d=True);assert sr==RATE
    result=np.zeros((round(SECONDS*RATE),2));count=min(len(data),len(result));result[:count]=data[:count]
    print('Rendered orchestral part:',role,flush=True);return role,result


def convolution_room(audio):
    rng=np.random.default_rng(SEED);t=np.arange(round(RATE*1.8))/RATE
    impulse=rng.normal(size=len(t))*np.exp(-t*4.5);impulse=lfilter([.12],[1,-.88],impulse)
    impulse[:round(.035*RATE)]=0;impulse/=np.sqrt(np.sum(impulse**2))
    return np.column_stack([fftconvolve(audio[:,i],impulse,mode='full')[:len(audio)] for i in range(2)])


def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.manual_seed(SEED)
    assert len(ROLES)==19 and max(CHANNELS.values())<=15
    write_json(OUT/'protocol.json',dict(seed=SEED,bars=BARS,bpm=BPM,roles=ROLES,model_channels=CHANNELS,
        trained=False,model='Original pretrained SkyTNT; separate neutral residual agents; no adapters loaded.',
        arrangement='Explicit staggered entries and legal velocity ranges; C-major pitch mask. Notes, rests, timing and durations are sampled, not copied or hand-scored.',
        sounds='General MIDI samples, including a choir patch. No real 60-person choir or modular hardware recording.',
        learning_claim='No learned orchestration or cooperation claim. All sampled notes are agent actions.',
        midi='19 role tracks split across two MIDI ports; per-role files also supplied. Piano agents share a model patch/channel but keep distinct track/role identities.'))
    model=nt.load_model();before=fingerprint(model);agents={r:oe.Musician().eval() for r in ROLES}
    episode=compose(model,agents);write_json(OUT/'episode.json',episode)
    # All requested parts participate in this order test by giving each a probe entry at zero.
    entries={r:s['entry'] for r,s in ROLES.items()}
    try:
        for s in ROLES.values():s['entry']=0
        a=compose(model,agents,1,SEED+99);b=compose(model,agents,1,SEED+99,order=list(reversed(ROLES)))
        assert a['notes']==b['notes']
    finally:
        for r,value in entries.items():ROLES[r]['entry']=value
    assert before==fingerprint(model);export_midis(episode)
    families={f:np.zeros((round(SECONDS*RATE),2)) for f in FAMILIES}
    family_gains=dict(organ=.75,pianos=.75,strings=.8,woodwinds=.38,brass=.35,synth=.25,choir=.35)
    with ThreadPoolExecutor(max_workers=3) as pool:
        for role,pcm in pool.map(render_role,ROLES):families[ROLES[role]['family']]+=pcm*family_gains[ROLES[role]['family']]
    dry=sum(families.values());wet=convolution_room(dry);mix=dry+.10*wet
    mix=np.tanh(mix*1.15)/1.15  # gentle saturation, without changing MIDI events
    fade=np.ones(len(mix));fade[-2*RATE:]=np.linspace(1,0,2*RATE);mix*=fade[:,None]
    sf.write(OUT/'orchestra_raw_mix.wav',mix,RATE,subtype='FLOAT');integrated,peak=loudness(OUT/'orchestra_raw_mix.wav')
    gain_db=min(-20-integrated,-2-peak);gain=10**(gain_db/20);checks={}
    for name,pcm in dict(orchestra=mix,**{f:pcm*fade[:,None] for f,pcm in families.items()}).items():
        wav=OUT/f'{name}.wav';mp3=OUT/f'{name}.mp3';sf.write(wav,pcm*gain,RATE,subtype='PCM_24')
        subprocess.run(['ffmpeg','-y','-v','error','-i',str(wav),'-q:a','2',str(mp3)],check=True)
        decoded=OUT/f'{name}_decoded.wav';subprocess.run(['ffmpeg','-y','-v','error','-i',str(mp3),str(decoded)],check=True)
        audio,sr=sf.read(decoded);decoded.unlink();assert np.isfinite(audio).all() and np.max(np.abs(audio))<1
        checks[name]=dict(seconds=len(audio)/sr,decoded_peak=float(np.max(np.abs(audio))),sha256=hashlib.sha256(mp3.read_bytes()).hexdigest())
    counts={r:sum(n[3]==s['id'] for n in episode['notes']) for r,s in ROLES.items()}
    write_json(OUT/'verification.json',dict(base_model_unchanged=True,base_sha256=before,all_notes_match_actions=True,
        all_19_roles_order_independent=True,midi_action_agreement=True,sampled_choir_not_live_singers=True,
        new_training=False,cooperation_established=False,generated_notes=counts,master_gain_db=gain_db,audio=checks))
    page(counts);print('Full orchestral prototype ready:',OUT/'listen.html',flush=True)


def page(counts):
    options='<option value="orchestra">Full orchestra</option>'+''.join(f'<option value="{f}">{f.title()} only</option>' for f in FAMILIES)
    (OUT/'listen.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,"><title>Full orchestral prototype</title><style>body{font:18px system-ui;background:#111923;color:#eaf0fa;max-width:950px;margin:35px auto;padding:20px;line-height:1.6}section{background:#203047;border-radius:14px;padding:24px;margin:24px 0}audio{width:100%;margin:20px 0}button,select{font:inherit;padding:10px}a{color:#a7d9ff}table{width:100%;border-collapse:collapse}td{border-bottom:1px solid #41526a;padding:6px}</style><h1>All the colours of the orchestra</h1><p>A new 39-second cinematic prototype: organ, strings, four pianos, woodwinds, brass, synth pad and sampled choir. The parts enter gradually.</p><section><label>Listen to <select id="part">'''+options+'''</select></label><audio id="audio" controls preload="metadata" src="orchestra.mp3"></audio><button id="play">Play from beginning</button><p>Family solos keep their level in the full mix; some are deliberately quiet.</p></section><p>The original pretrained model generates each part. The entry schedule, C-major pitch mask and dynamics limits are arrangement controls; this is not a newly trained orchestral ensemble.</p><p>The choir is a sampled ensemble patch, not a recording of 60 human singers. Synth and room effects are digital render processing. General MIDI samples approximate the acoustic sections.</p><details><summary>Which parts actually played?</summary><table>'''+''.join(f'<tr><td>{r.replace("_"," ").title()}</td><td>{n} notes</td></tr>' for r,n in counts.items())+'''</table></details><p><a href="verification.json">Audio and action checks</a> · <a href="protocol.json">Generation details</a> · <a href="orchestra_multiport.mid">Multiport orchestral MIDI</a></p><script>const a=document.getElementById('audio');document.getElementById('part').onchange=e=>{a.pause();a.src=e.target.value+'.mp3';a.load()};document.getElementById('play').onclick=()=>{a.currentTime=0;a.play().catch(()=>{})};</script></html>''',encoding='utf-8')


if __name__=='__main__':main()
