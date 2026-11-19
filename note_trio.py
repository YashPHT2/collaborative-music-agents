"""Three independently learning note-level musicians over frozen musical knowledge.

No phrase bank, chord progression, supplied theme, note-copy controller or energy
curve. All exported notes are committed policy actions. Instrument/range limits,
quantization, initial key, tempo and maximum length are environment constraints.
"""
import argparse
import copy
import hashlib
import json
import math
import random
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import mido
import numpy as np
import torch
from band import BandSampler, Musician, TeamCritic, fingerprint, load_model, update
from generate_piano import ROOT, render
from evaluate_baseline import validate_audio

TPB=480
ROLES={
    'organ':dict(channel=0,track=1,patch=19,pitches=list(range(48,89)),cap=4,polyphony=4),
    'strings':dict(channel=1,track=2,patch=48,pitches=list(range(55,92)),cap=2,polyphony=4),
    'cello':dict(channel=2,track=3,patch=42,pitches=list(range(36,68)),cap=2,polyphony=1),
}
FORMAT=1


@dataclass
class Brief:
    bars:int=4
    bpm:int=100
    tonic:int=4
    minor:bool=True
    def __post_init__(self):
        if not 1<=self.bars<=32 or not 40<=self.bpm<=180 or not 0<=self.tonic<12:
            raise ValueError('Invalid trio brief')


def visible(ledger,beat,role=None):
    return [copy.deepcopy(n) for n in ledger if n[1]<beat*TPB
            and (n[1]>=(beat-32)*TPB or n[1]<8*TPB)
            and (role is None or n[3]==ROLES[role]['channel'])]


def state_features(brief,beat,history):
    if any(n[1]>=beat*TPB for n in history): raise ValueError('Future/current notes in observation')
    x=torch.zeros(32)
    recent=[n for n in history if n[1]>=(beat-8)*TPB]
    # Distinct bins for all twelve observed pitch classes, no modulo-8 aliasing.
    total=sum(n[2] for n in recent)
    for n in recent: x[n[4]%12]+=n[2]/max(1,total)
    x[12+brief.tonic]=1
    x[24]=beat/(brief.bars*4); x[25]=brief.bpm/180; x[26]=brief.minor
    x[27]=math.sin(beat*math.pi/2); x[28]=math.cos(beat*math.pi/2)
    for i,s in enumerate(ROLES.values()): x[29+i]=min(1,sum(n[3]==s['channel'] for n in recent)/24)
    return x


class NoteSampler(BandSampler):
    def __init__(self,model,role,seed,brief,beat,history):
        super().__init__(model,role,seed,brief,beat,history,roles=ROLES)
        self.exploration=.02

    def onsets(self):
        return [step for step in range(16) if self.beat*TPB+step*30>=self.last_tick
                and sum(n[3]==self.settings['channel'] and n[1]<=self.beat*TPB+step*30<n[1]+n[2]
                        for n in self.observed)<self.settings['polyphony']]

    def constrain_values(self,kind,values,decoded):
        if kind=='velocity': return list(range(32,97,4))
        if kind=='duration':
            maximum=min(64,(self.brief.bars*4*TPB-(self.beat*TPB+decoded['time2']*30))//30)
            return list(range(2,maximum+1,2)) or [maximum]
        return values


def commit(ledger,actions,beat,brief):
    """Validate the whole simultaneous joint action before committing anything."""
    if set(actions)!=set(ROLES): raise ValueError('All three roles must propose')
    additions=[]
    for r,s in ROLES.items():
        ns=actions[r]
        if len(ns)>s['cap']: raise ValueError('Event cap exceeded')
        for n in ns:
            if not (n[0]=='note' and beat*TPB<=n[1]<(beat+1)*TPB and n[1]%30==0
                    and n[2]>0 and n[2]%30==0 and n[1]+n[2]<=brief.bars*4*TPB
                    and n[3]==s['channel'] and n[4] in s['pitches'] and 1<=n[5]<=127):
                raise ValueError(f'Invalid {r} note')
        own=[n for n in ledger+ns if n[3]==s['channel']]
        for n in ns:
            active=[p for p in own if p[1]<=n[1]<p[1]+p[2]]
            if len(active)>s['polyphony'] or len({p[4] for p in active})!=len(active):
                raise ValueError('Polyphony/repeated-pitch overlap')
        additions.extend(copy.deepcopy(ns))
    return sorted(ledger+additions,key=lambda n:(n[1],n[3],n[4]))


def reward(ledger,end):
    """Causal, bounded engineering proxy; never presented as musical taste."""
    notes=[n for n in ledger if n[1]<end]
    left=max(0,end-1920); active=[n for n in notes if n[1]+n[2]>left]
    if not active: return -1.,dict(silence=1.)
    harmony=weight=0.
    for i,a in enumerate(active):
        for b in active[i+1:]:
            if a[3]==b[3]: continue
            overlap=max(0,min(end,a[1]+a[2],b[1]+b[2])-max(left,a[1],b[1]))
            interval=abs(a[4]-b[4])%12
            harmony+=overlap*(1. if interval in [0,3,4,5,7,8,9] else .2)
            weight+=overlap
    harmony/=max(1,weight)
    densities=[]; participation=[]; motion=[]; repetition=0.
    for r,s in ROLES.items():
        own=sorted([n for n in notes if n[3]==s['channel'] and n[1]>=end-7680],key=lambda n:n[1])
        onset=sum(left<=n[1]<end for n in own)/4
        # Broad density bands permit sustained notes and rests rather than a fixed pulse.
        upper={'organ':2.5,'strings':1.,'cello':1.25}[r]
        densities.append(1-min(1,max(0,onset-upper)/upper))
        participation.append(min(1,sum(n[2] for n in own)/max(1,min(end,7680))/.35))
        line=[max(n[4] for n in own if n[1]==t) for t in sorted({n[1] for n in own})]
        if len(line)>1: motion.append(np.mean([math.exp(-max(0,abs(a-b)-7)/7) for a,b in zip(line,line[1:])]))
        if len(line)>=8:
            dominance=max(line.count(p) for p in set(line))/len(line)
            repetition+=max(0,(dominance-.8)/.2)/3
    # Return of self-generated interval fragments; useful variation is retained.
    lead=[n[4] for n in notes if n[3]==0]
    ints=[b-a for a,b in zip(lead,lead[1:])]
    motifs=[tuple(ints[i:i+3]) for i in range(len(ints)-2)]
    recurrence=0.
    if len(motifs)>8:
        old=set(motifs[:-4]); recurrence=sum(m in old and len(set(m))>1 for m in motifs[-4:])/4
    parts=dict(harmony=harmony,density=float(np.mean(densities)),participation=float(np.mean(participation)),
               voice_leading=float(np.mean(motion)) if motion else 0.,motif=recurrence,degeneracy=repetition)
    value=.3*parts['harmony']+.2*parts['density']+.15*parts['participation']+.2*parts['voice_leading']+.15*recurrence-repetition
    return float(np.clip(value,-1,1)),parts


def rollout(model,agents,critic,brief,seed,shared=True,order=None):
    ledger=[]; steps=[]; trace=[]; decisions=[]
    for beat in range(brief.bars*4):
        state=state_features(brief,beat,visible(ledger,beat))
        with torch.no_grad(): value=float(critic(state))
        actions={}; records={}
        for role in (order or list(ROLES)):
            past=visible(ledger,beat,None if shared else role)
            sampler=NoteSampler(model,role,seed+beat*1009+list(ROLES).index(role)*100003,brief,beat,past)
            role_state=state_features(brief,beat,past)
            actions[role],records[role]=sampler.propose(agents[role],role_state)
            # These proposals themselves become the output; no arranger expands or copies them.
            for n in actions[role]: decisions.append(dict(role=role,beat=beat,note=copy.deepcopy(n)))
            trace.append(dict(beat=beat,role=role,history_hash=hashlib.sha256(json.dumps(past).encode()).hexdigest(),
                              partner_notes=sum(n[3]!=ROLES[role]['channel'] for n in past),
                              emitted_notes=len(actions[role]),decisions=len(records[role])))
        ledger=commit(ledger,actions,beat,brief)
        score,parts=reward(ledger,(beat+1)*TPB) if beat%4==3 else (0.,{})
        steps.append(dict(state=state,value=value,reward=score,components=parts,records=records))
    assert sorted(d['note'] for d in decisions)==sorted(ledger)
    return dict(notes=ledger,steps=steps,trace=trace,decisions=decisions,brief=asdict(brief),seed=seed)


def checkpoint(path,agents,critic,opts,co,initial,logs,settings,elapsed):
    payload=dict(format_version=FORMAT,roles=ROLES,actors={r:a.state_dict() for r,a in agents.items()},
                 critic=critic.state_dict(),optimizers={r:o.state_dict() for r,o in opts.items()},critic_optimizer=co.state_dict(),
                 initial=initial,logs=logs,settings=settings,elapsed=elapsed,torch_rng=torch.get_rng_state(),
                 cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                 numpy_rng=np.random.get_state(),python_rng=random.getstate())
    temp=path.with_suffix('.tmp'); torch.save(payload,temp); temp.replace(path)


def save_episode(root,name,ep):
    for suffix,data in [('events',ep['notes']),('trace',ep['trace']),('decisions',ep['decisions'])]:
        (root/f'{name}_{suffix}.json').write_text(json.dumps(data,indent=2))


def tempo_seconds(bars,bpm=100,tempo_map=None):
    changes=tempo_map or [(0,bpm)]
    if not changes or changes[0][0]!=0 or any(not 40<=v<=180 for t,v in changes):raise ValueError('Invalid tempo map')
    if any(t<0 or t>=bars*1920 for t,v in changes) or any(changes[i][0]>=changes[i+1][0] for i in range(len(changes)-1)):
        raise ValueError('Tempo changes must be ordered and within the score')
    return sum(((changes[i+1][0] if i+1<len(changes) else bars*1920)-tick)/480*round(60_000_000/value)/1_000_000 for i,(tick,value) in enumerate(changes))


def midi_file(path,notes,role=None,bpm=100,tempo_map=None):
    midi=mido.MidiFile(ticks_per_beat=480); meta=mido.MidiTrack(); midi.tracks.append(meta)
    meta.append(mido.MetaMessage('time_signature',numerator=4,denominator=4))
    last=0
    for tick,value in (tempo_map or [(0,bpm)]):
        if tick<last or not 40<=value<=180:raise ValueError('Invalid tempo map')
        meta.append(mido.MetaMessage('set_tempo',tempo=round(60_000_000/value),time=tick-last));last=tick
    for r,s in ROLES.items():
        if role is not None and r!=role: continue
        tr=mido.MidiTrack(); midi.tracks.append(tr); ch=s['channel']
        tr.append(mido.MetaMessage('track_name',name=r)); tr.append(mido.Message('program_change',channel=ch,program=s['patch']))
        tr.append(mido.Message('control_change',channel=ch,control=7,value={'organ':127,'strings':64,'cello':80}[r]))
        events=[]
        for n in notes:
            if n[3]!=ch: continue
            events.extend([(n[1],1,mido.Message('note_on',channel=ch,note=n[4],velocity=n[5])),
                           (n[1]+n[2],0,mido.Message('note_off',channel=ch,note=n[4],velocity=0))])
        last=0
        for tick,_,msg in sorted(events,key=lambda e:(e[0],e[1])): msg.time=tick-last; tr.append(msg); last=tick
    midi.save(path)
    tick=0; active={}; actual=[]
    for msg in mido.merge_tracks(mido.MidiFile(path).tracks):
        tick+=msg.time
        if msg.type=='note_on' and msg.velocity:
            key=msg.channel,msg.note
            assert key not in active
            active[key]=tick,msg.velocity
        elif msg.type=='note_off' or (msg.type=='note_on' and not msg.velocity):
            t,v=active.pop((msg.channel,msg.note)); actual.append(['note',t,tick-t,msg.channel,msg.note,v])
    assert not active and sorted(actual)==sorted(n for n in notes if role is None or n[3]==ROLES[role]['channel'])


def export(root,name,ep,stems=True,allow_silent_stems=False):
    save_episode(root,name,ep); checks={}
    for stem in (['mix',*ROLES] if stems else ['mix']):
        path=root/f'{name}_{stem}.mid'; midi_file(path,ep['notes'],None if stem=='mix' else stem,ep['brief']['bpm'],ep.get('tempo_map'))
        wav=path.with_suffix('.wav'); render(path,wav); tmp=wav.with_name(wav.stem+'_master.wav')
        seconds=tempo_seconds(ep['brief']['bars'],ep['brief']['bpm'],ep.get('tempo_map'))+2.4
        filters=f'apad,atrim=duration={seconds},afade=t=out:st={seconds-1}:d=1,loudnorm=I={-18 if stem=="mix" else -22}:TP=-1.5:LRA=11'
        if ep.get('room_effect'):filters='aecho=0.8:0.7:173|311|487:0.18|0.12|0.07,'+filters
        subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(wav),'-af',filters,'-ar','44100','-c:a','pcm_s16le',str(tmp)],check=True)
        tmp.replace(wav)
        subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(wav),'-q:a','2',str(wav.with_suffix('.mp3'))],check=True)
        expected_notes=[n for n in ep['notes'] if stem=='mix' or n[3]==ROLES[stem]['channel']]
        if allow_silent_stems and not expected_notes:
            import wave
            with wave.open(str(wav),'rb') as stream:
                pcm=np.frombuffer(stream.readframes(stream.getnframes()),dtype='<i2').astype(float)/32768
                duration=stream.getnframes()/stream.getframerate()
            peak=float(np.max(np.abs(pcm))) if len(pcm) else 0.
            if peak>=.001:raise ValueError('Unexpected sound in an empty MIDI stem')
            checks[stem]=dict(seconds=duration,peak=peak,silent=True,expected_notes=0)
        else:
            checks[stem]=validate_audio(wav)
    return checks


def write_page(root):
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Three note-generating musicians</title><style>body{font:18px system-ui;background:#101724;color:#e8edf5;max-width:850px;margin:45px auto;padding:20px}p{line-height:1.6}button,select{font:inherit;padding:10px;margin:5px;background:#263850;color:white;border:1px solid #65748b;border-radius:8px}audio{width:100%;margin:20px 0}.card{padding:22px;background:#1c293d;border-radius:14px}a{color:#9bd2ff}</style><h1>Three note-generating musicians</h1><p>Organ, upper strings and cello. Every note is a policy action. This first joint-training result has no supplied motif, chord sequence, phrase bank or programmed build. Musical quality is still under evaluation.</p><div class="card"><label>Performance <select id="method"><option value="trained">After joint training</option><option value="initial">Matched untrained baseline</option><option value="no_partner">Trained without partner observations</option></select></label><label>Part <select id="stem"><option value="mix">Full trio</option><option value="organ">Organ</option><option value="strings">Upper strings</option><option value="cello">Cello</option></select></label><audio id="audio" controls src="trained_mix.mp3"></audio><p id="message">Solo levels are adjusted for listening.</p><button id="start">Play from beginning</button></div><p><a href="trained_mix.mid">Download trained MIDI</a> · <a href="verification.json">Run verification</a></p><p>The three independently trained residual policies adapt a shared frozen pretrained MIDI model. They select note/rest, onset, pitch, velocity and duration. All see committed past beats, propose privately, and commit together. The reward is an engineering proxy, not a learned measure of musical enjoyment.</p><script>const a=document.getElementById('audio'),m=document.getElementById('method'),s=document.getElementById('stem');function swap(){s.disabled=m.value!=='trained';if(s.disabled)s.value='mix';const t=a.currentTime;a.pause();a.src=m.value+'_'+s.value+'.mp3';a.addEventListener('loadedmetadata',()=>{a.currentTime=Math.min(t,a.duration||0)},{once:true});a.load()}m.onchange=swap;s.onchange=swap;document.getElementById('start').onclick=()=>{a.currentTime=0;a.play().catch(()=>{})};</script></html>'''
    (root/'listen.html').write_text(page,encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--updates',type=int,default=8)
    p.add_argument('--episodes',type=int,default=4); p.add_argument('--max-minutes',type=float,default=30)
    p.add_argument('--seed',type=int,default=20261101); p.add_argument('--resume',action='store_true')
    p.add_argument('--output',type=Path,default=ROOT/'outputs'/'note_trio_v1'); args=p.parse_args()
    if args.updates<1 or args.episodes<2 or not 0<args.max_minutes<=480: p.error('Invalid training bounds')
    root=args.output; root.mkdir(parents=True,exist_ok=True)
    if (root/'latest.pt').exists() and not args.resume: raise ValueError('Use --resume or a new --output')
    torch.set_num_threads(4); torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed%2**32)
    agents={r:Musician() for r in ROLES}; critic=TeamCritic()
    opts={r:torch.optim.Adam(a.parameters(),lr=1e-4) for r,a in agents.items()}; co=torch.optim.Adam(critic.parameters(),lr=3e-4)
    initial={r:copy.deepcopy(a.state_dict()) for r,a in agents.items()}; logs=[]; elapsed=0
    settings=dict(seed=args.seed,episodes=args.episodes,model_revision='0f8f265d4330f4e46527ac2313200254c5757f5f',sampler='note-trio-v1',reward='pitched-ensemble-v1')
    if args.resume:
        saved=torch.load(root/'latest.pt',map_location='cpu',weights_only=False)
        if saved['format_version']!=FORMAT or saved['roles']!=ROLES or saved['settings']!=settings: raise ValueError('Incompatible resume settings')
        for r,a in agents.items(): a.load_state_dict(saved['actors'][r]); opts[r].load_state_dict(saved['optimizers'][r])
        critic.load_state_dict(saved['critic']); co.load_state_dict(saved['critic_optimizer'])
        initial,logs,elapsed=saved['initial'],saved['logs'],saved['elapsed']
        torch.set_rng_state(saved['torch_rng']); np.random.set_state(saved['numpy_rng']); random.setstate(saved['python_rng'])
        if saved['cuda_rng'] and torch.cuda.is_available(): torch.cuda.set_rng_state_all(saved['cuda_rng'])
    model=load_model(); base=fingerprint(model); started=time.perf_counter()
    for u in range(len(logs),args.updates):
        bars=4 if u<4 else 8 if u<16 else 16
        episodes=[]; tick=time.perf_counter()
        for e in range(args.episodes):
            brief=Brief(bars,100,[4,9,2,7][(u*args.episodes+e)%4],True)
            ep=rollout(model,agents,critic,brief,1100000+u*args.episodes+e)
            episodes.append(ep); print(f'Note PPO {u+1}/{args.updates}, episode {e+1}/{args.episodes}: {len(ep["notes"])} policy-generated notes',flush=True)
        metrics=update(agents,critic,opts,co,episodes)
        logs.append(dict(update=u+1,bars=bars,episodes=args.episodes,seconds=time.perf_counter()-tick,
                         mean_bar_reward=float(np.mean([s['reward'] for ep in episodes for s in ep['steps'] if s['components']])),learning=metrics))
        checkpoint(root/'latest.pt',agents,critic,opts,co,initial,logs,settings,elapsed+time.perf_counter()-started)
        (root/'training_log.json').write_text(json.dumps(logs,indent=2)); print(json.dumps(logs[-1]),flush=True)
        if time.perf_counter()-started>=args.max_minutes*60 or elapsed+time.perf_counter()-started>=8*3600: break
    baseline={r:Musician() for r in ROLES}
    for r,a in baseline.items(): a.load_state_dict(initial[r])
    # Validation prompts are distinct from training and the final listening seed.
    validation=[]
    for seed in [7201001,7201002]:
        for name,policies in [('initial',baseline),('trained',agents)]:
            ep=rollout(model,policies,critic,Brief(4,100,seed%12,True),seed)
            validation.append(dict(seed=seed,method=name,mean_bar_reward=float(np.mean([s['reward'] for s in ep['steps'] if s['components']]))))
    (root/'validation.json').write_text(json.dumps(validation,indent=2))
    checks={}; counts={}; provenance={}
    for name,policies,shared in [('trained',agents,True),('initial',baseline,True),('no_partner',agents,False)]:
        ep=rollout(model,policies,critic,Brief(24,100,4,True),9301001,shared=shared)
        checks[name]=export(root,name,ep,stems=name=='trained')
        counts[name]={r:sum(n[3]==s['channel'] for n in ep['notes']) for r,s in ROLES.items()}
        provenance[name]=dict(decision_notes=len(ep['decisions']),exported_notes=len(ep['notes']),
            note_sha256=hashlib.sha256(json.dumps(sorted(ep['notes'])).encode()).hexdigest(),
            sampled_attributes={r:sorted({int(record['observation'][1056:1064].argmax())
                for step in ep['steps'] for record in step['records'][r]}) for r in ROLES})
        print(f'Exported {name}: {counts[name]}',flush=True)
    before={r:fingerprint(a) for r,a in baseline.items()}
    verification=dict(format_version=FORMAT,roles=ROLES,completed_updates=len(logs),training_episodes=sum(x['episodes'] for x in logs),
                      actors_changed={r:fingerprint(a)!=before[r] for r,a in agents.items()},
                      frozen_backbone_unchanged=fingerprint(model)==base,all_output_notes_are_committed_policy_actions=True,
                      agent_actions=['note_or_rest','onset','pitch','velocity','duration'],
                      scripted_melody=False,chord_progression=False,motif_copying=False,phrase_templates=False,programmed_build=False,
                      limits='Instrument programs/ranges/polyphony, event safety caps, grid, initial key, tempo, maximum length',
                      architecture='Three separately trained note-level residual policies over one frozen pretrained MIDI Transformer; joint PPO and central critic',
                      learned_quality_advantage_established=False,counts=counts,exports=checks,provenance=provenance,settings=settings)
    assert all(verification['actors_changed'].values()) and verification['frozen_backbone_unchanged']
    (root/'verification.json').write_text(json.dumps(verification,indent=2)); write_page(root)
    (root/'REPORT.md').write_text('# Three note-generating musicians\n\nEvery output note comes directly from a musician policy proposal. '
        'Each independent residual actor selects event/rest, onset, pitch, velocity and duration over frozen pretrained musical distributions. '
        'Proposals are simultaneous from committed past history. There is no supplied melody, chord sequence, phrase lookup, copy controller or energy curve. '
        'Joint PPO uses shared bounded heuristic rewards and a central training-only critic. '
        'Weight changes, sensitivity and heuristic scores do not establish musical enjoyment or a learned collaboration advantage. '
        'Initial, trained and partner-hidden controls share seed, masks, renderer and loudness treatment. '
        'All decisions are in *_decisions.json and are checked against the exact exported MIDI.\n',encoding='utf-8')
    print(json.dumps(verification,indent=2),flush=True)


if __name__=='__main__': main()
