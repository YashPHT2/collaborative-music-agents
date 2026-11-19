"""Two pretrained note agents, authentic felt samples, and a shared-piano audition."""
import contextlib
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

import mido
import numpy as np
import requests
from scipy.signal import resample_poly
import soundfile as sf
import torch

import note_trio as nt
import offline_ensemble as oe
from band import fingerprint
from audit_projected_roles import loudness
from train_ensemble_repair import write_json

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/focus_duet_v1'
ASSET=ROOT/'assets/meta_felt_piano'
REPO='facebookincubator/Meta-open-sound'
SEED=9900101
BPM=70
BARS=8
RATE=48000
SECONDS=32*round(60_000_000/BPM)/1_000_000+3
SCALE={0,2,4,5,7,9,11}
DUET=dict(piano=dict(channel=0,track=1,patch=0,pitches=[p for p in range(48,85) if p%12 in SCALE],cap=2,polyphony=4),
          cello=dict(channel=1,track=2,patch=42,pitches=[p for p in range(43,61) if p%12 in SCALE],cap=1,polyphony=1))


@contextlib.contextmanager
def duet_roles():
    previous=nt.ROLES;nt.ROLES=copy.deepcopy(DUET)
    try:yield
    finally:nt.ROLES=previous


class FocusSampler(nt.NoteSampler):
    def onsets(self):
        steps=[0,8] if self.role=='piano' else [0]
        return [s for s in steps if self.beat*480+s*30>=self.last_tick and
                sum(n[3]==self.settings['channel'] and n[1]<=self.beat*480+s*30<n[1]+n[2]
                    for n in self.observed)<self.settings['polyphony']]

    def constrain_values(self,kind,values,decoded):
        if kind=='velocity':return list(range(24,53,4)) if self.role=='piano' else list(range(28,53,4))
        if kind=='duration':
            maximum=min(64 if self.role=='piano' else 128,
                        (self.brief.bars*1920-self.beat*480-decoded['time2']*30)//30)
            minimum=8 if self.role=='piano' else 32
            return list(range(min(minimum,maximum),maximum+1,2))
        return values


@torch.no_grad()
def compose(model,agents,bars=BARS,seed=SEED,order=None):
    brief=nt.Brief(bars,BPM,0,False);ledger=[];decisions=[];trace=[]
    with duet_roles():
        for beat in range(bars*4):
            past=nt.visible(ledger,beat);actions={}
            for role in order or DUET:
                sampler=FocusSampler(model,role,seed+beat*1009+list(DUET).index(role)*100003,brief,beat,past)
                actions[role],_=sampler.propose(agents[role],nt.state_features(brief,beat,past))
                decisions.extend(dict(role=role,beat=beat,note=n) for n in actions[role])
            ledger=nt.commit(ledger,actions,beat,brief)
            trace.append(dict(beat=beat,history_sha256=hashlib.sha256(json.dumps(past).encode()).hexdigest(),
                              emitted={r:len(v) for r,v in actions.items()}))
    assert sorted(ledger)==sorted(d['note'] for d in decisions)
    return dict(seed=seed,brief=nt.asdict(brief),notes=ledger,decisions=decisions,trace=trace)


def midi(path,notes):
    result=mido.MidiFile(ticks_per_beat=480);meta=mido.MidiTrack();result.tracks.append(meta)
    meta.append(mido.MetaMessage('set_tempo',tempo=round(60_000_000/BPM)))
    meta.append(mido.MetaMessage('time_signature',numerator=4,denominator=4))
    meta.append(mido.MetaMessage('key_signature',key='C'))
    for role,settings in DUET.items():
        track=mido.MidiTrack();result.tracks.append(track);channel=settings['channel']
        track.append(mido.MetaMessage('track_name',name=role))
        track.append(mido.Message('program_change',channel=channel,program=settings['patch']))
        events=[]
        for n in notes:
            if n[3]!=channel:continue
            events.extend([(n[1],1,mido.Message('note_on',channel=channel,note=n[4],velocity=n[5])),
                           (n[1]+n[2],0,mido.Message('note_off',channel=channel,note=n[4],velocity=0))])
        previous=0
        for tick,_,message in sorted(events,key=lambda x:(x[0],x[1])):message.time=tick-previous;track.append(message);previous=tick
    result.save(path)
    active={};actual=[];tick=0
    for message in mido.merge_tracks(mido.MidiFile(path).tracks):
        tick+=message.time
        if message.type=='note_on' and message.velocity:active[message.channel,message.note]=(tick,message.velocity)
        elif message.type=='note_off' or message.type=='note_on' and not message.velocity:
            start,velocity=active.pop((message.channel,message.note));actual.append(['note',start,tick-start,message.channel,message.note,velocity])
    assert not active and sorted(actual)==sorted(notes)


def sample_assets(notes):
    ASSET.mkdir(parents=True,exist_ok=True);pin=ASSET/'source.json'
    if pin.exists():source=json.loads(pin.read_text())
    else:
        response=requests.get(f'https://api.github.com/repos/{REPO}/commits/main',timeout=30);response.raise_for_status()
        source=dict(repository=REPO,commit=response.json()['sha'],license='CC0-1.0',format='Published OGG samples',
                    instrument='Felt-damped upright piano',release='meta-piano-v1.0.0');write_json(pin,source)
    base=f'https://raw.githubusercontent.com/{REPO}/{source["commit"]}/'
    for name,remote in [('LICENSE','LICENSE'),('README.md','systems/instruments/meta-piano/README.md'),
                         ('meta-piano.sfz','systems/instruments/meta-piano/meta-piano.sfz')]:
        if not (ASSET/name).exists():
            response=requests.get(base+remote,timeout=30);response.raise_for_status();(ASSET/name).write_bytes(response.content)
    names=['C','Cs','D','Ds','E','F','Fs','G','Gs','A','As','B'];paths={}
    for pitch in sorted({n[4] for n in notes}):
        for layer in (1,2):
            filename=f'INSPiano_Vel0{layer}_{names[pitch%12]}{pitch//12-1}_01.ogg'
            paths[pitch,layer]=ASSET/filename
    def fetch(path):
        if not path.exists():
            response=requests.get(base+'systems/instruments/meta-piano/ogg/'+path.name,timeout=60);response.raise_for_status()
            temporary=path.with_suffix('.tmp');temporary.write_bytes(response.content);temporary.replace(path)
        return dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    with ThreadPoolExecutor(max_workers=4) as pool:checks=list(pool.map(fetch,paths.values()))
    write_json(OUT/'piano_sample_sources.json',dict(**source,samples=checks,library_license_file=str(ASSET/'LICENSE')))
    return paths


def layer_weights(velocity):
    if velocity<=24:return 1.,0.
    if velocity>=39:return 0.,1.
    phase=(velocity-24)/15*math.pi/2
    return math.cos(phase),math.sin(phase)


def render_piano(notes,paths):
    output=np.zeros((round(SECONDS*RATE),2),dtype=np.float64);cache={}
    for n in notes:
        start=round(n[1]/480*round(60_000_000/BPM)/1_000_000*RATE)
        hold=round(n[2]/480*round(60_000_000/BPM)/1_000_000*RATE);release=round(.25*RATE)
        for layer,weight in enumerate(layer_weights(n[5]),1):
            if weight<1e-9:continue
            if (n[4],layer) not in cache:
                pcm,sr=sf.read(paths[n[4],layer],always_2d=True)
                if pcm.shape[1]==1:pcm=np.repeat(pcm,2,axis=1)
                assert pcm.shape[1]==2
                if sr!=RATE:pcm=resample_poly(pcm,RATE//math.gcd(RATE,sr),sr//math.gcd(RATE,sr))
                cache[n[4],layer]=pcm
            pcm=cache[n[4],layer];length=min(len(pcm),hold+release,len(output)-start)
            sound=pcm[:length].copy();ramp=min(length,round(.003*RATE));sound[:ramp]*=np.linspace(0,1,ramp)[:,None]
            if length>hold:sound[hold:]*=np.linspace(1,0,length-hold)[:,None]
            output[start:start+length]+=sound*weight
    return output


def render_cello(notes):
    midi(OUT/'cello.mid',notes)
    fluid=next((ROOT/'tools/fluidsynth').rglob('fluidsynth.exe'))
    subprocess.run([str(fluid),'-ni','-F',str(OUT/'cello_raw.wav'),'-r',str(RATE),'-g','0.6',
                    str(ROOT/'assets/soundfont.sf2'),str(OUT/'cello.mid')],check=True,capture_output=True)
    pcm,sr=sf.read(OUT/'cello_raw.wav',always_2d=True);assert sr==RATE
    output=np.zeros((round(SECONDS*RATE),2));length=min(len(pcm),len(output));output[:length]=pcm[:length]
    return output


def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.manual_seed(SEED)
    protocol=dict(seed=SEED,bpm=BPM,bars=BARS,key='C major',roles=DUET,
        training='No new training or fine-tuned adapters; original pretrained model with constrained sampling.',
        constraints='Major-scale pitch mask, role ranges, soft velocity ranges, piano eighth-note grid, cello beat grid and longer legal cello durations. No forced notes or fixed progression.',
        comparison='Identical piano waveform; cello is muted only in piano-alone version.',
        piano_render='Authentic Meta Piano OGG samples, pp/mp equal-power velocity blending, 250ms note-off release.',
        focus_or_mood_benefit='Unmeasured; requires listening and study-task feedback.')
    write_json(OUT/'protocol.json',protocol)
    model=nt.load_model();before=fingerprint(model)
    agents={r:oe.Musician().eval() for r in DUET}
    episode=compose(model,agents);write_json(OUT/'episode.json',episode)
    print('Generated new duet:',{r:sum(n[3]==s['channel'] for n in episode['notes']) for r,s in DUET.items()},flush=True)
    a=compose(model,agents,2,SEED+99);b=compose(model,agents,2,SEED+99,order=list(reversed(DUET)))
    assert a['notes']==b['notes'] and before==fingerprint(model)
    piano=[n for n in episode['notes'] if n[3]==0];cello=[n for n in episode['notes'] if n[3]==1]
    if not piano:raise ValueError('Piano policy chose silence; retained episode, no artificial notes added')
    midi(OUT/'duet.mid',episode['notes']);midi(OUT/'piano_alone.mid',piano)
    paths=sample_assets(piano);print('Authentic felt samples ready:',len(paths),flush=True)
    p=render_piano(piano,paths);c=render_cello(cello)
    piano_rms=float(np.sqrt(np.mean(p*p)));cello_rms=float(np.sqrt(np.mean(c*c)))
    cello_gain=min(1.,piano_rms*10**(-16/20)/max(cello_rms,1e-12));c*=cello_gain
    fade=round(RATE);envelope=np.ones(len(p));envelope[-fade:]=np.linspace(1,0,fade)
    p*=envelope[:,None];c*=envelope[:,None]
    variants={'piano_alone':p,'duet':p+c};levels={}
    for name,pcm in variants.items():sf.write(OUT/f'{name}_raw.wav',pcm,RATE,subtype='FLOAT');levels[name]=loudness(OUT/f'{name}_raw.wav')
    # Common gain preserves the exact same piano amplitude, so only cello differs.
    gain_db=min(-22-levels['piano_alone'][0],*[-2-peak for integrated,peak in levels.values()])
    gain=10**(gain_db/20);checks={}
    for name,pcm in variants.items():
        wav=OUT/f'{name}.wav';mp3=OUT/f'{name}.mp3';sf.write(wav,pcm*gain,RATE,subtype='PCM_24')
        subprocess.run(['ffmpeg','-y','-v','error','-i',str(wav),'-q:a','2',str(mp3)],check=True)
        decoded=OUT/f'{name}_decoded.wav';subprocess.run(['ffmpeg','-y','-v','error','-i',str(mp3),str(decoded)],check=True)
        audio,sr=sf.read(decoded);decoded.unlink();assert np.isfinite(audio).all() and np.max(np.abs(audio))<1
        checks[name]=dict(seconds=len(audio)/sr,measured_lufs=loudness(wav)[0],decoded_peak=float(np.max(np.abs(audio))),
                          sha256=hashlib.sha256(mp3.read_bytes()).hexdigest())
    assert np.allclose((variants['duet']-variants['piano_alone']),c)
    write_json(OUT/'verification.json',dict(base_model_unchanged=True,base_sha256=before,no_new_training=True,
        all_notes_match_agent_actions=True,midi_action_agreement=True,agent_order_independent=True,
        identical_piano_in_both_versions=True,same_master_gain_db=gain_db,cello_gain=cello_gain,
        cello_to_piano_rms_db=None if cello_rms==0 else 20*math.log10(cello_rms*cello_gain/piano_rms),
        generated_notes={r:sum(n[3]==s['channel'] for n in episode['notes']) for r,s in DUET.items()},audio=checks))
    page();print('Focus listening prototype ready:',OUT/'listen.html',flush=True)


def page():
    (OUT/'listen.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,"><title>Felt piano — focus prototype</title><style>body{font:18px system-ui;line-height:1.6;background:#111b21;color:#edf3ed;max-width:850px;margin:35px auto;padding:20px}section{background:#203235;padding:22px;border-radius:14px;margin:24px 0}audio{width:100%}button,select{font:inherit;padding:10px;margin:10px 0}label{display:block}a{color:#b3e7cf}</style><h1>A quieter direction</h1><p>About 30 seconds of newly generated music at 70 BPM. Compare the exact same felt-piano performance with and without quiet cello.</p><p>The original pretrained model generated these notes with a soft C-major sampling preset. No new focus-specific training was performed.</p><section><h2>1. Felt piano alone</h2><audio controls preload="metadata" src="piano_alone.mp3"></audio><button>Play from beginning</button></section><section><h2>2. Felt piano + quiet cello</h2><audio controls preload="metadata" src="duet.mp3"></audio><button>Play from beginning</button></section><label>Which would you study with? <select id="choice"><option value="">Choose</option><option value="piano_alone">Piano alone</option><option value="duet">Piano + cello</option><option value="neither">Neither</option><option value="same">Both equally</option></select></label><label>Does it still feel scary? <select id="scary"><option value="">Choose</option><option>Yes</option><option>A little</option><option>No</option></select></label><button id="save">Download my feedback</button><p id="status" role="status"></p><p>Real felt-damped piano samples from <a href="https://github.com/facebookincubator/Meta-open-sound/releases/tag/meta-piano-v1.0.0">Meta Open Sound</a> (CC0). This is a sound-and-generation prototype; concentration benefits have not been tested.</p><p><a href="protocol.json">Generation details</a> · <a href="verification.json">Audio checks</a> · <a href="duet.mid">Duet MIDI</a></p><script>for(const s of document.querySelectorAll('section')){const a=s.querySelector('audio');s.querySelector('button').onclick=()=>{a.currentTime=0;a.play().catch(()=>{})};a.addEventListener('play',()=>{for(const other of document.querySelectorAll('audio'))if(a!==other)other.pause()})}document.getElementById('save').onclick=()=>{const choice=document.getElementById('choice').value,scary=document.getElementById('scary').value;if(!choice||!scary){document.getElementById('status').textContent='Please answer both questions.';return}const url=URL.createObjectURL(new Blob([JSON.stringify({schema:'focus-duet-v1',seed:9900101,choice,scary},null,2)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download='focus-duet-feedback.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};</script></html>''',encoding='utf-8')


if __name__=='__main__':main()
