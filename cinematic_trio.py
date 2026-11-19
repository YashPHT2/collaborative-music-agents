"""Original cinematic trio: separate phrase policies, explicit arranging grammar.

The frozen MIDI model proposes the motif. PPO learns bounded phrase choices;
voicing, pulse, chord sequence and formal build are engineered, not emergent.
"""
import argparse
import copy
import hashlib
import json
import math
import subprocess
from pathlib import Path

import mido
import numpy as np
import torch
from torch import nn

from band import load_model, fingerprint
from accompany import EventSampler
from generate_piano import ROOT, render
from evaluate_baseline import validate_audio
from ppo_adapter import gae

TPB, BAR, BPM = 480, 1920, 100
OUT = ROOT / 'outputs' / 'cinematic_trio_v1'
ROLES = {'organ': (0, 19), 'strings': (1, 48), 'cello': (2, 42)}
CHORDS = [(4, [4,7,11,2]), (0,[0,4,7,11]), (7,[7,11,2,9]), (2,[2,6,9,4]),
          (9,[9,0,4,7]), (0,[0,4,7,11]), (11,[11,3,6,9]), (4,[4,7,11,2])]
OBS, ACTIONS = 40, 6


class PhraseAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(OBS,64), nn.Tanh(), nn.Linear(64,64), nn.Tanh())
        self.head = nn.Linear(64,ACTIONS)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def distribution(self, observation, reference):
        return torch.distributions.Categorical(logits=reference.log()+2*torch.tanh(self.head(self.body(observation))))


def energy(bar, bars):
    phase=bar/max(1,bars-1)
    return float(np.interp(phase,[0,.18,.4,.68,.82,1],[.18,.32,.55,.9,1,.12]))


def observation(ledger,bar,bars):
    """Only previously committed bars, including all twelve pitch classes."""
    start=bar*BAR
    if any(n[1]>=start for n in ledger): raise ValueError('Future/current notes leaked into observation')
    x=torch.zeros(OBS); x[0]=bar/max(1,bars); x[1]=energy(bar,bars)
    root,pcs=CHORDS[bar%8]; x[2+root]=1
    for pc in pcs: x[14+pc]=1
    for i,(role,(channel,_)) in enumerate(ROLES.items()):
        own=[n for n in ledger if n[3]==channel and start-2*BAR<=n[1]<start]
        x[26+i]=min(1,len(own)/16)
        x[29+i]=sum(n[5] for n in own)/max(1,len(own))/127
        x[32+i]=(own[-1][4]/127) if own else 0
    recent=[n for n in ledger if start-BAR<=n[1]<start]
    x[35]=sum(n[2] for n in recent)/(BAR*8)
    x[36]=math.sin(bar*math.pi/4); x[37]=math.cos(bar*math.pi/4)
    x[38]=float(bar>=bars-2); x[39]=1
    return x


@torch.inference_mode()
def propose_theme(model, seed):
    sampler=EventSampler(model,seed,100,.85)
    sampler.rows=sampler.rows[:1]
    for e in [['time_signature',0,0,0,3,1],['key_signature',0,0,0,8,1],
              ['set_tempo',0,0,0,100],['patch_change',0,0,1,0,19]]:
        sampler.rows.append(sampler.tok.event2tokens(e))
    pool=[64,67,71,74,76,79,83]
    pitches=[]
    from generate_piano import DynamicCache
    for i in range(16):
        hidden=model(torch.tensor([sampler.rows],device=model.device))[:,-1]
        cache=DynamicCache(); row=None
        values=[None,0,0,1,0,None,64,8]
        for j,kind in enumerate(['event','time1','time2','track','channel','pitch','velocity','duration']):
            logits=model.forward_token(hidden if j==0 else None,None if row is None else row[:,-1:],cache=cache)[0,-1].float()
            if kind=='pitch':
                ids=[sampler.tok.parameter_ids[kind][p] for p in pool]
                score=logits[ids]; score=(score-score.mean())/score.std(unbiased=False).clamp_min(.8)
                previous=pitches[-1] if pitches else 71
                score+=torch.tensor([-.18*abs(p-previous)-1.5*(p==previous) for p in pool],device=model.device)
                pitch=pool[torch.multinomial(torch.softmax(score,0),1,generator=sampler.generator).item()]
                token=sampler.tok.parameter_ids[kind][pitch]
            else:
                token=sampler.tok.event_ids['note'] if j==0 else sampler.tok.parameter_ids[kind][values[j]]
            t=torch.tensor([[token]],device=model.device); row=t if row is None else torch.cat([row,t],1)
        pitches.append(pitch)
        sampler.append_note(['note',i*240,210,0,pitch,64],1)
    return [pool.index(p) for p in pitches], pitches


def note(tick,duration,ch,pitch,velocity):
    return ['note',int(tick),int(duration),ch,int(pitch),int(np.clip(velocity,1,110))]


def realize(role, action, theme, ledger, bar, bars):
    """Independent bar proposals; shared chord/form grammar is intentional."""
    channel,_=ROLES[role]; start=bar*BAR; e=energy(bar,bars)
    root,pcs=CHORDS[bar%8]
    own=[n for n in ledger if n[3]==channel]
    last=own[-1][4] if own else {'organ':71,'strings':67,'cello':40}[role]
    partner=[n for n in ledger if n[3]!=channel and n[1]>=start-BAR]
    partner_velocity=np.mean([n[5] for n in partner]) if partner else 50
    base={'organ':48,'strings':34,'cello':45}[role]+e*{'organ':25,'strings':23,'cello':22}[role]
    velocity=base+np.clip((partner_velocity-55)*.1,-3,3)+(action%3-1)*3
    if role=='organ':
        pool=[p for p in range(64,85) if p%12 in pcs]
        motif=theme[(bar%2)*8:(bar%2+1)*8]
        # The motif and its transformations are explicit arrangement scaffolding.
        variants=[motif,motif[2:]+motif[:2],motif[:4]+motif[2:6],
                  [min(6,v+1) for v in motif],motif[:6]+list(reversed(motif[4:6])),motif]
        motif=variants[action]
        result=[]
        for i,degree in enumerate(motif):
            p=pool[min(len(pool)-1,degree)]
            result.append(note(start+i*240,225,channel,p,velocity+(3 if i%4==0 else -2)))
        if bar==bars-1:
            result=result[:4]+[note(start+960,960,channel,76,velocity-7),note(start+960,960,channel,71,velocity-12)]
        return result
    if role=='strings':
        # Choose an inversion by voice-leading, with independently learned variation.
        ranges=[(60,79),(64,83),(57,76),(60,79),(64,83),(60,79)]
        lo,hi=ranges[action]
        chord_pcs=pcs[:3] if bar==bars-1 else pcs
        pitches=[]
        for pc in chord_pcs[1:]:
            choices=[p for p in range(lo,hi+1) if p%12==pc]
            pitches.append(min(choices,key=lambda p:abs(p-last)))
        if bar<2: pitches=pitches[:1]
        else: pitches=pitches[:2 if e<.6 else 3]
        return [note(start,BAR,channel,p,velocity-(3 if i else 0)) for i,p in enumerate(sorted(set(pitches)))]
    pool=[p for p in range(36,55) if p%12==root]
    bass=min(pool,key=lambda p:abs(p-last))
    fifth=next(p for p in range(bass,bass+13) if p%12==(root+7)%12)
    if bar==bars-1: return [note(start,BAR,channel,bass,velocity-8)]
    rhythm=([0,960] if e<.55 else [0,480,960,1440])
    if action==1 and e>.4: rhythm=[0,720,960,1680]
    if action==4 and bar>1: rhythm=[0,960,1440]
    duration=min(900,min(b-a for a,b in zip(rhythm,rhythm[1:]))-30)
    return [note(start+t,min(duration,BAR-t),channel,bass if i%2==0 or action in [0,2] else fifth,velocity-(i%2)*5)
            for i,t in enumerate(rhythm)]


def score_bar(ledger,new,bar,bars):
    pcs=CHORDS[bar%8][1]
    fit=np.mean([n[4]%12 in pcs for n in new])
    organ=[n for n in new if n[3]==0]
    leaps=np.mean([min(1,abs(a[4]-b[4])/12) for a,b in zip(organ,organ[1:])]) if len(organ)>1 else 1
    recent=[n for n in ledger if n[1]>=(bar-1)*BAR]
    # Role-to-role dynamics and audible continuity, bounded engineering proxies.
    levels=[np.mean([n[5] for n in new if n[3]==ch]) for ch,_ in ROLES.values()]
    balance=1-min(1,(max(levels)-min(levels))/40)
    continuity=1.
    for ch,_ in ROLES.values():
        before=[n for n in recent if n[3]==ch]; after=[n for n in new if n[3]==ch]
        if before and after: continuity-=min(1,abs(before[-1][4]-after[0][4])/24)/3
    return float(.25*fit+.3*(1-leaps)+.25*balance+.2*continuity)


def rollout(agents,critic,theme,seed,bars=8,order=None,shared=True,greedy=False):
    rng={r:torch.Generator().manual_seed(seed+i*100003) for i,r in enumerate(ROLES)}
    ledger=[]; steps=[]; trace=[]
    for bar in range(bars):
        state=observation(ledger,bar,bars)
        actions={}; records={}; proposals={}
        with torch.no_grad(): value=float(critic(state).squeeze())
        for r in (order or ROLES):
            observed=ledger if shared else [n for n in ledger if n[3]==ROLES[r][0]]
            obs=observation(observed,bar,bars)
            ref=torch.tensor([.28,.1,.18,.1,.12,.22])
            with torch.no_grad(): dist=agents[r].distribution(obs,ref)
            a=int(dist.probs.argmax()) if greedy else int(torch.multinomial(dist.probs,1,generator=rng[r]))
            actions[r]=a; records[r]=dict(obs=obs,ref=ref,action=a,logp=float(dist.log_prob(torch.tensor(a))))
            proposals[r]=realize(r,a,theme,observed,bar,bars)
        new=sorted([n for r in ROLES for n in proposals[r]],key=lambda n:(n[1],n[3],n[4]))
        reward=score_bar(ledger,new,bar,bars)
        trace.append(dict(bar=bar+1,actions=actions,energy=energy(bar,bars),chord=CHORDS[bar%8][1],
                          history_hash=hashlib.sha256(json.dumps(ledger).encode()).hexdigest(),reward=reward))
        steps.append(dict(state=state,value=value,reward=reward,records=records))
        ledger.extend(new); ledger.sort(key=lambda n:(n[1],n[3],n[4]))
    return dict(notes=ledger,steps=steps,trace=trace)


def train(agents,critic,theme,updates=24):
    opts={r:torch.optim.Adam(a.parameters(),lr=3e-4) for r,a in agents.items()}
    co=torch.optim.Adam(critic.parameters(),lr=5e-4); logs=[]
    for u in range(updates):
        episodes=[rollout(agents,critic,theme,510000+u*8+j,bars=8 if u<12 else 16) for j in range(8)]
        samples=[]; aa=[]; rr=[]
        for ep in episodes:
            adv,ret=gae(torch.tensor([s['reward'] for s in ep['steps']]),torch.tensor([s['value'] for s in ep['steps']]))
            samples.extend(ep['steps']); aa.extend(adv); rr.extend(ret)
        adv=torch.stack(aa); adv=(adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-6)
        returns=torch.stack(rr); losses={}
        for role,agent in agents.items():
            records=[s['records'][role] for s in samples]
            obs=torch.stack([r['obs'] for r in records]); ref=torch.stack([r['ref'] for r in records])
            old=torch.tensor([r['logp'] for r in records]); actions=torch.tensor([r['action'] for r in records])
            for epoch in range(4):
                d=agent.distribution(obs,ref); logp=d.log_prob(actions); ratio=(logp-old).exp()
                kl=(d.probs*(d.logits-ref.log())).sum(-1).mean()
                loss=-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean()+.05*kl-.01*d.entropy().mean()
                oldkl=float(((ratio-1)-(logp-old)).mean().detach())
                if epoch and oldkl>.02: break
                opts[role].zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(agent.parameters(),.5); opts[role].step()
            losses[role]=float(loss.detach())
        states=torch.stack([s['state'] for s in samples])
        for _ in range(4):
            loss=(critic(states).squeeze(-1)-returns).square().mean()
            co.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(critic.parameters(),.5); co.step()
        logs.append(dict(update=u+1,mean_reward=float(np.mean([s['reward'] for s in samples])),actor_loss=losses))
        if (u+1)%4==0: print(f'Joint phrase PPO {u+1}/{updates}: reward {logs[-1]["mean_reward"]:.3f}',flush=True)
    torch.save(dict(format_version=1,roles=ROLES,actors={r:a.state_dict() for r,a in agents.items()},
                    critic=critic.state_dict(),logs=logs,theme=theme),OUT/'phrase_agents.pt')
    return logs


def write_midi(path, notes, role=None):
    midi=mido.MidiFile(ticks_per_beat=TPB)
    conductor=mido.MidiTrack(); midi.tracks.append(conductor)
    conductor.append(mido.MetaMessage('set_tempo',tempo=600000))
    conductor.append(mido.MetaMessage('time_signature',numerator=4,denominator=4))
    for r,(ch,patch) in ROLES.items():
        if role and r!=role: continue
        track=mido.MidiTrack(); midi.tracks.append(track)
        track.append(mido.MetaMessage('track_name',name=r))
        track.append(mido.Message('program_change',channel=ch,program=patch))
        track.append(mido.Message('control_change',channel=ch,control=7,value={'organ':127,'strings':64,'cello':80}[r]))
        # Expression curves are engineered orchestration and retained in every stem.
        events=[]
        for bar in range(24):
            for k in range(4):
                t=bar*BAR+k*480
                expr=int(70+30*energy(bar,24))
                events.append((t,1,mido.Message('control_change',channel=ch,control=11,value=expr)))
        for n in notes:
            if n[3]!=ch: continue
            events.extend([(n[1],2,mido.Message('note_on',channel=ch,note=n[4],velocity=n[5])),
                           (n[1]+n[2],0,mido.Message('note_off',channel=ch,note=n[4],velocity=0))])
        last=0
        for t,_,msg in sorted(events,key=lambda v:(v[0],v[1])):
            msg.time=t-last; track.append(msg); last=t
    midi.save(path)


def verify_midi(path,notes,role=None):
    active={}; decoded=[]; tick=0
    for msg in mido.merge_tracks(mido.MidiFile(path).tracks):
        tick+=msg.time
        if msg.type=='note_on' and msg.velocity:
            key=(msg.channel,msg.note)
            assert key not in active, 'Repeated-pitch overlap'
            active[key]=(tick,msg.velocity)
        elif msg.type=='note_off' or (msg.type=='note_on' and not msg.velocity):
            t,v=active.pop((msg.channel,msg.note)); decoded.append(note(t,tick-t,msg.channel,msg.note,v))
    expected=[n for n in notes if role is None or n[3]==ROLES[role][0]]
    assert not active and sorted(decoded)==sorted(expected)
    return len(decoded)


def export(episode):
    checks={}
    for stem in ['mix',*ROLES]:
        path=OUT/f'first_light_{stem}.mid'; write_midi(path,episode['notes'],None if stem=='mix' else stem)
        count=verify_midi(path,episode['notes'],None if stem=='mix' else stem)
        wav=path.with_suffix('.wav'); render(path,wav)
        temp=wav.with_name(wav.stem+'_master.wav')
        filters='apad,atrim=duration=60,afade=t=in:d=0.12,afade=t=out:st=57.6:d=2.4'
        filters+=',loudnorm=I=-18:TP=-1.5:LRA=11' if stem=='mix' else ',loudnorm=I=-22:TP=-2:LRA=11'
        subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(wav),'-af',filters,'-ar','44100','-c:a','pcm_s16le',str(temp)],check=True)
        temp.replace(wav)
        subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(wav),'-q:a','2',str(path.with_suffix('.mp3'))],check=True)
        checks[stem]=dict(notes=count,audio=validate_audio(wav),loudness_target_lufs=-18 if stem=='mix' else -22)
    return checks


def page():
    (OUT/'listen.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>First Light — cinematic trio</title><style>body{background:#0d1421;color:#e8eaf0;font:18px system-ui;max-width:780px;margin:70px auto;padding:24px}h1{font-size:48px}p{line-height:1.6;color:#bec9d9}audio{width:100%;margin:24px 0}button,select{padding:12px;background:#24344d;color:white;border:1px solid #536480;border-radius:8px;font:inherit;margin:5px}a{color:#a4ceff}.card{padding:22px;background:#182336;border-radius:16px}small{line-height:1.6;display:block;color:#aebbd0}</style>
<p>ORIGINAL CINEMATIC STUDY · 60 SECONDS</p><h1>First Light</h1><p>A repeating organ motif, rising upper strings, and a cello pulse. Three independently trained phrase policies over a shared arranged harmonic form.</p>
<div class="card"><label for="part">Listen to </label><select id="part"><option value="mix">Full trio</option><option value="organ">Organ</option><option value="strings">Upper strings</option><option value="cello">Cello</option></select>
<audio id="audio" controls preload="metadata" src="first_light_mix.mp3"></audio><button id="begin">From the beginning</button><button id="build">Hear the build · 28.8s</button>
<p>0–19s · Establishing the theme<br>19–38s · Gathering motion<br>38–50s · Full ensemble<br>50–60s · Return and release</p><small>Solo playback levels are adjusted for audibility; they differ from the part levels in the full mix.</small></div>
<p><a href="first_light_mix.mid">Download MIDI</a> · <a href="first_light_mix.mp3" download>Download MP3</a></p>
<details><summary>How this performance was made</summary><p>The pretrained MIDI model proposed the seed motif. Three new phrase policies received a short joint PPO pilot. Their choices control motif variations, string inversions, cello patterns and dynamics. Chords, rhythmic templates, theme reprises, expression curves and the formal build are explicitly arranged. This does not demonstrate learned orchestration or a proven quality improvement. Existing General MIDI sounds are used.</p></details>
<script>const a=document.getElementById('audio');document.getElementById('part').onchange=e=>{const t=a.currentTime,play=!a.paused;a.pause();a.src='first_light_'+e.target.value+'.mp3';a.addEventListener('loadedmetadata',()=>{a.currentTime=Math.min(t,a.duration||0);if(play)a.play().catch(()=>{})},{once:true});a.load()};function jump(t){if(a.readyState){a.currentTime=t;a.play().catch(()=>{})}else{a.addEventListener('loadedmetadata',()=>{a.currentTime=t;a.play().catch(()=>{})},{once:true});a.load()}}document.getElementById('begin').onclick=()=>jump(0);document.getElementById('build').onclick=()=>jump(28.8);</script></html>''',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,default=20261031); p.add_argument('--updates',type=int,default=24); args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True); torch.set_num_threads(4); torch.manual_seed(20261031)
    model=load_model(); base=fingerprint(model)
    theme_path=OUT/'theme.json'
    if theme_path.exists(): theme_info=json.loads(theme_path.read_text()); theme=theme_info['ranks']
    else:
        theme,pitches=propose_theme(model,args.seed)
        theme_info=dict(seed=args.seed,ranks=theme,model_proposed_pitches=pitches,source='Frozen SkyTNT constrained organ pitch sampling; no commercial melody supplied')
        theme_path.write_text(json.dumps(theme_info,indent=2))
    assert fingerprint(model)==base
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    agents={r:PhraseAgent() for r in ROLES}; initial=copy.deepcopy(agents)
    before={r:fingerprint(a) for r,a in agents.items()}
    critic=nn.Sequential(nn.Linear(OBS,64),nn.Tanh(),nn.Linear(64,1))
    logs=train(agents,critic,theme,args.updates)
    episode=rollout(agents,critic,theme,args.seed+123,bars=24)
    matched=rollout(initial,critic,theme,args.seed+123,bars=24)
    independent=rollout(agents,critic,theme,args.seed+123,bars=24,shared=False)
    (OUT/'events.json').write_text(json.dumps(episode['notes'],indent=2))
    (OUT/'trace.json').write_text(json.dumps(episode['trace'],indent=2))
    write_midi(OUT/'initial_control.mid',matched['notes']); write_midi(OUT/'without_partners.mid',independent['notes'])
    checks=export(episode); page()
    results=dict(title='First Light',seconds=60,seed=args.seed,roles=ROLES,
                 architecture='Three separate bar-level residual phrase policies; model-proposed motif plus arranged event expansion',
                 agents_changed={r:fingerprint(a)!=before[r] for r,a in agents.items()},
                 backbone_unchanged=True,training_episodes=args.updates*8,
                 score='Bounded engineered harmony/voice-leading/dynamic-balance proxy',
                 arrangement='Eight-chord form, motif transformations, rhythmic templates and energy/expression curves are engineered',
                 instrument_specific_transformer_finetuning=False,quality_improvement_established=False,
                 partner_sensitivity={r:[n for n in episode['notes'] if n[3]==s[0]]!=[n for n in independent['notes'] if n[3]==s[0]] for r,s in ROLES.items()},
                 exports=checks,theme=theme_info)
    assert all(results['agents_changed'].values())
    (OUT/'verification.json').write_text(json.dumps(results,indent=2))
    (OUT/'REPORT.md').write_text('# First Light\n\nAn original 60-second cinematic organ, upper-strings and cello study. No commercial melody or recording was supplied.\n\n'
        'The MIDI foundation model proposed the seed pitches. New, separate phrase policies were trained jointly with PPO. '
        'This is a hierarchical phrase-policy pilot; it does not fine-tune the instrument Transformers. '
        'Harmony, timing templates, motif transformations and expressive build are programmed arrangement. '
        'Training reward and changed policy weights do not establish musical quality or emergent cooperation. '
        'The existing SoundFont provides GM church organ, strings and cello.\n',encoding='utf-8')
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__': main()
