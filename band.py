"""Three independently trainable MIDI musicians and a simultaneous beat environment."""
import copy
import math
import hashlib
import json
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
import torch
from torch import nn
from generate_piano import ROOT, MIDIModel, DynamicCache
from accompany import EventSampler, TPB, write_result

ROLES = {
    'piano': dict(channel=0, track=1, patch=0, pitches=list(range(48,89)), cap=6, polyphony=3),
    'bass': dict(channel=1, track=2, patch=32, pitches=list(range(28,56)), cap=2, polyphony=1),
    'drums': dict(channel=9, track=3, patch=0, pitches=[36,38,42,46,45,49], cap=8, polyphony=6),
}
KINDS = ['event','time1','time2','track','channel','pitch','velocity','duration']
OBS = 1072  # 1024 frozen features + 32 state + 8 token-type + 8 partial-event

@dataclass
class Brief:
    bars: int = 4
    bpm: int = 100
    tonic: int = 9
    minor: bool = True
    reward_version: str = 'band-heuristic-v1'

    def __post_init__(self):
        if not 1 <= self.bars <= 16 or not 30 <= self.bpm <= 240 or not 0 <= self.tonic < 12:
            raise ValueError('Expected 1..16 bars, 30..240 BPM, tonic 0..11')
        if self.reward_version not in ['band-heuristic-v1','band-heuristic-v2']:
            raise ValueError('Unknown reward version')

    def energy(self, beat):
        phase = beat / (self.bars*4)
        return .25 + .65*min(phase/.75,1) if phase < .875 else .35

def state_features(brief, beat, ledger):
    x = torch.zeros(32)
    x[brief.tonic] = 1
    x[12] = brief.minor
    x[13] = brief.bpm/240
    x[14] = beat/(brief.bars*4)
    x[15] = brief.energy(beat)
    x[16:18] = torch.tensor([math.sin(beat*math.pi/2),math.cos(beat*math.pi/2)])
    for i,settings in enumerate(ROLES.values()):
        recent = [n for n in ledger if n[3]==settings['channel'] and n[1]>=(beat-4)*TPB]
        x[18+i] = min(len(recent)/24,1)
        x[21+i] = sum(n[2] for n in recent)/(16*TPB)
    for n in ledger[-32:]:
        if n[3]!=9:
            x[24+n[4]%8] += 1/32
    return x

class Musician(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(OBS,256),nn.Tanh(),nn.Linear(256,256),nn.Tanh())
        self.head = nn.Linear(256,128)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def distribution(self, observation, reference):
        residual = 2*torch.tanh(self.head(self.encoder(observation)))
        logits = torch.where(reference>0,reference.clamp_min(1e-30).log()+residual,
                             torch.full_like(reference,-torch.inf))
        return torch.distributions.Categorical(logits=logits)

class TeamCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(32,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,1))

    def forward(self,x):
        return self.net(x).squeeze(-1)

def fingerprint(module):
    h=hashlib.sha256()
    for name,p in module.state_dict().items():
        h.update(name.encode()); h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def visible_history(ledger, beat, role, shared=True):
    channel=ROLES[role]['channel']
    return [copy.deepcopy(n) for n in ledger
            if n[1]<beat*TPB and (n[1]>=max(0,beat-32)*TPB or n[1]<8*TPB)
            and (shared or n[3]==channel)]

class BandSampler(EventSampler):
    def constrain_values(self,kind,values,decoded):
        return values

    def adjust_logits(self,kind,values,logits,decoded):
        return logits

    def __init__(self,model,role,seed,brief,beat,history,roles=None):
        super().__init__(model,seed,brief.bpm,.9)
        self.roles=ROLES if roles is None else roles
        self.role,self.settings,self.brief,self.beat=role,self.roles[role],brief,beat
        # MIDI tokenizer key signature stores circle-of-fifths index shifted by 7.
        key_sf = next(sf for sf in range(-7,8) if (7*sf+(9 if brief.minor else 0))%12==brief.tonic)
        self.rows=self.rows[:1]
        for e in [['time_signature',0,0,0,3,1],['key_signature',0,0,0,key_sf+7,int(brief.minor)],
                  ['set_tempo',0,0,0,brief.bpm]]:
            self.rows.append(self.tok.event2tokens(e))
        for s in self.roles.values():
            self.rows.append(self.tok.event2tokens(['patch_change',0,0,s['track'],s['channel'],s['patch']]))
        self.observed=copy.deepcopy(history)
        for n in sorted(history,key=lambda n:(n[1],n[3],n[4])):
            track=next(s['track'] for s in self.roles.values() if s['channel']==n[3])
            self.append_note(n,track)

    def onsets(self):
        result=[]
        for step in [0,4,8,12]:
            tick=self.beat*TPB+step*30
            active=[n for n in self.observed if n[3]==self.settings['channel'] and n[1]<=tick<n[1]+n[2]]
            if tick>=self.last_tick and len(active)<self.settings['polyphony']:
                result.append(step)
        return result

    def legal_parameters(self,kind,decoded):
        """Shared sampling/teacher-forcing support; musical decisions stay legal."""
        if kind=='event':return [0,1],[self.tok.event_ids['note'],self.tok.eos_id]
        if kind=='time1':values=[self.beat-self.last_tick//TPB]
        elif kind=='time2':values=self.onsets()
        elif kind=='track':values=[self.settings['track']]
        elif kind=='channel':values=[self.settings['channel']]
        elif kind=='pitch':
            tick=self.beat*TPB+decoded['time2']*30
            active={n[4] for n in self.observed if n[3]==self.settings['channel'] and n[1]<=tick<n[1]+n[2]}
            values=[p for p in self.settings['pitches'] if p not in active]
        elif kind=='velocity':values=list(range(40,101,4))
        elif kind=='duration':
            tick=self.beat*TPB+decoded['time2']*30
            maximum=min(64,(self.brief.bars*4*TPB-tick)//30)
            values=[4] if self.role=='drums' else list(range(4,maximum+1,4))
        else:raise ValueError(kind)
        values=self.constrain_values(kind,values,decoded)
        return values,[self.tok.parameter_ids[kind][v] for v in values]

    @torch.no_grad()
    def propose(self,policy,state):
        notes,records=[],[]
        for _ in range(self.settings['cap']):
            onsets=self.onsets()
            if not onsets:
                break
            x=torch.tensor([self.rows[self.cached_length:]],device=self.model.device)
            hidden=self.model(x,cache=self.cache)[:,-1]
            self.cached_length=len(self.rows)
            token_cache=DynamicCache()
            row=None
            decoded={}
            for index in range(8):
                kind=KINDS[index]
                values,ids=self.legal_parameters(kind,decoded)
                if not ids: raise RuntimeError(f'Empty support: {self.role}/{kind}')
                logits=self.model.forward_token(hidden if index==0 else None,
                            None if row is None else row[:,-1:],cache=token_cache)[0,-1].float()
                adjusted=self.adjust_logits(kind,values,logits[ids],decoded)
                if len(ids)==1:
                    value=values[0]; token=ids[0]
                else:
                    reference=torch.zeros(getattr(policy,'action_size',128))
                    reference[values]=torch.softmax(adjusted/.9,dim=-1).cpu()
                    # Same exploration mixture for trained and untrained controls.
                    exploration=getattr(self,'exploration',.01)
                    reference[values]=(1-exploration)*reference[values]+exploration/len(values)
                    obs=torch.zeros(OBS)
                    obs[:1024]=hidden[0].float().cpu()
                    obs[1024:1056]=state
                    obs[1056+index]=1
                    if row is not None:
                        obs[1064:1064+row.shape[1]]=row[0].float().cpu()/self.tok.vocab_size
                    policy_device=next(policy.parameters()).device
                    distribution=policy.distribution(obs.to(policy_device),reference.to(policy_device))
                    # Each musician owns its RNG, independent of evaluation order.
                    value=torch.multinomial(distribution.probs.to(self.model.device),1,generator=self.generator).item()
                    token=ids[values.index(value)]
                    records.append(dict(observation=obs,reference=reference,action=value,
                                        logp=float(distribution.log_prob(torch.tensor(value,device=policy_device)))))
                decoded[kind]=value
                token_tensor=torch.tensor([[token]],device=self.model.device)
                row=token_tensor if row is None else torch.cat([row,token_tensor],dim=1)
                if index==0 and value==1:
                    return notes,records
            note=['note',self.beat*TPB+decoded['time2']*30,decoded['duration']*30,
                  self.settings['channel'],decoded['pitch'],decoded['velocity']]
            notes.append(note); self.observed.append(note)
            self.append_note(note,self.settings['track'])
        return notes,records

def ensemble_reward(ledger,start,end,brief):
    """Bounded heuristic diagnostics, not a learned measure of human taste."""
    active=[n for n in ledger if n[1]<end and n[1]+n[2]>start]
    piano=[n for n in active if n[3]==0]; bass=[n for n in active if n[3]==1]
    pairs=[(a,b,max(0,min(end,a[1]+a[2],b[1]+b[2])-max(start,a[1],b[1]))) for a in piano for b in bass]
    weight=sum(w for a,b,w in pairs)
    harmony=sum(w*(1 if (a[4]-b[4])%12 in [0,3,4,5,7,8,9] else -.5) for a,b,w in pairs)/max(1,weight)
    attacks=[n for n in ledger if start<=n[1]<end]
    low={n[1] for n in attacks if n[3]==1}
    kick={n[1] for n in attacks if n[3]==9 and n[4]==36}
    groove=len(low&kick)/max(1,len(low|kick))
    energy=brief.energy(start/TPB)
    density=len(attacks)/((end-start)/TPB)
    arrangement=1-min(1,abs(density-(1+4*energy))/(2+4*energy)) if active else 0
    # Return of an earlier pitch-interval motif, with an allowance for variation.
    current=[n[4] for n in attacks if n[3]==0]
    past=[n[4] for n in ledger if n[3]==0 and n[1]<start][-32:]
    def intervals(xs): return [b-a for a,b in zip(xs,xs[1:])]
    ci,pi=intervals(current),intervals(past)
    phrase=0
    if len(ci)>=2 and len(pi)>=2:
        old={tuple(pi[i:i+2]) for i in range(len(pi)-1)}
        phrase=sum(tuple(ci[i:i+2]) in old for i in range(len(ci)-1))/max(1,len(ci)-1)
        if len(set(current))==1: phrase=0
    penalty=1 if not active else 0
    for channel in [0,1]:
        pitches=[n[4] for n in ledger if n[3]==channel and start-4*TPB<=n[1]<end]
        if len(pitches)>=8:
            penalty+=max(0,(max(pitches.count(p) for p in set(pitches))/len(pitches)-.75)*4)
    penalty+=min(1,max(0,density-10)/6)
    components=dict(harmony=harmony,groove=groove,phrase=phrase,arrangement=arrangement,degeneracy=penalty)
    return .25*(harmony+groove+phrase+arrangement)-penalty,components

class BandEnv:
    def reset(self,brief,seed):
        self.brief,self.seed,self.beat,self.ledger=brief,seed,0,[]
        return state_features(brief,0,[])

    def step(self,actions_by_agent):
        if set(actions_by_agent)!=set(ROLES): raise ValueError('All three proposals are required')
        if self.beat>=self.brief.bars*4: raise ValueError('Episode is already complete')
        start,end=self.beat*TPB,(self.beat+1)*TPB
        additions=[]
        for role in ROLES:
            s=ROLES[role]
            if len(actions_by_agent[role])>s['cap']: raise ValueError('Beat event cap exceeded')
            for n in actions_by_agent[role]:
                if not (n[0]=='note' and start<=n[1]<end and n[3]==s['channel'] and n[4] in s['pitches']
                        and n[1]%120==0 and n[2]>0 and n[2]%120==0 and 1<=n[5]<=127
                        and n[1]+n[2]<=self.brief.bars*4*TPB):
                    raise ValueError(f'Invalid {role} proposal')
                additions.append(copy.deepcopy(n))
            own=[n for n in self.ledger+additions if n[3]==s['channel']]
            for n in actions_by_agent[role]:
                sounding=[p for p in own if p[1]<=n[1]<p[1]+p[2]]
                if len(sounding)>s['polyphony'] or len({p[4] for p in sounding})!=len(sounding):
                    raise ValueError('Polyphony or repeated-pitch overlap violated')
        self.ledger.extend(additions)
        self.ledger.sort(key=lambda n:(n[1],n[3],n[4]))
        self.beat+=1
        # Sparse musical rewards have fixed bar windows, not future information.
        reward_fn=ensemble_reward
        if self.brief.reward_version=='band-heuristic-v2':
            from band_reward import ensemble_reward_v2
            reward_fn=ensemble_reward_v2
        reward,parts=reward_fn(self.ledger,end-4*TPB,end,self.brief) if self.beat%4==0 else (0,{})
        return state_features(self.brief,self.beat,self.ledger),reward,self.beat==self.brief.bars*4,parts

def rollout(model,policies,critic,brief,seed,shared=True,order=None):
    env=BandEnv(); state=env.reset(brief,seed)
    steps=[]; trace=[]
    for beat in range(brief.bars*4):
        actions,records={},{}
        with torch.no_grad(): value=float(critic(state))
        for role in (order or list(ROLES)):
            history=visible_history(env.ledger,beat,role,shared)
            sampler=BandSampler(model,role,seed+beat*1009+list(ROLES).index(role)*100003,brief,beat,history)
            actions[role],records[role]=sampler.propose(policies[role],state if shared else state_features(brief,beat,history))
            trace.append(dict(beat=beat,role=role,history_hash=hashlib.sha256(json.dumps(history).encode()).hexdigest(),
                              partner_events=sum(n[3]!=ROLES[role]['channel'] for n in history)))
        next_state,reward,done,parts=env.step(actions)
        steps.append(dict(state=state,value=value,reward=reward,records=records,components=parts))
        state=next_state
    return dict(steps=steps,notes=env.ledger,trace=trace,brief=asdict(brief),seed=seed)

def update(policies,critic,optimizers,critic_optimizer,episodes,auxiliary=None):
    from ppo_adapter import gae
    samples=[]
    for episode in episodes:
        steps=episode['steps']
        adv,ret=gae(torch.tensor([s['reward'] for s in steps]),torch.tensor([s['value'] for s in steps]))
        samples.extend((s,a,t) for s,a,t in zip(steps,adv,ret))
    advantages=torch.stack([s[1] for s in samples])
    advantages=(advantages-advantages.mean())/advantages.std(unbiased=False).clamp_min(1e-6)
    returns=torch.stack([s[2] for s in samples]); states=torch.stack([s[0].get('critic_state',s[0]['state']) for s in samples])
    logs={}
    for role,actor in policies.items():
        rows=[]; groups=[]; old=[]; targets=[]
        for i,(step,_,_) in enumerate(samples):
            rs=step['records'][role]
            if rs:
                group=len(old); groups.extend([group]*len(rs)); rows.extend(rs)
                old.append(sum(r['logp'] for r in rs)); targets.append(advantages[i])
        if not rows: raise RuntimeError(f'No trainable decisions for {role}')
        observations=torch.stack([r['observation'] for r in rows]); refs=torch.stack([r['reference'] for r in rows])
        actions=torch.tensor([r['action'] for r in rows]); groups=torch.tensor(groups)
        old=torch.tensor(old); targets=torch.stack(targets)
        for epoch in range(4):
            # Complete beat actions remain together. Chunk forward passes to bound RAM.
            logps=[]; entropies=[]; kls=[]
            for ix in torch.arange(len(rows)).split(256):
                dist=actor.distribution(observations[ix],refs[ix])
                logps.append(dist.log_prob(actions[ix])); entropies.append(dist.entropy())
                kls.append((dist.probs*(dist.logits.clamp_min(-1e9)-refs[ix].clamp_min(1e-30).log())).sum(-1))
            logp=torch.zeros(len(old)).index_add(0,groups,torch.cat(logps))
            ratio=(logp-old).exp(); delta=logp-old
            loss=-torch.minimum(ratio*targets,ratio.clamp(.8,1.2)*targets).mean()+.05*torch.cat(kls).mean()-.005*torch.cat(entropies).mean()
            auxiliary_loss=auxiliary(role,actor) if auxiliary is not None else None
            if auxiliary_loss is not None:loss=loss+auxiliary_loss
            if not torch.isfinite(loss): raise RuntimeError('Non-finite actor loss')
            kl=float(((ratio-1)-delta).mean().detach())
            if epoch>0 and kl>.02: break
            optimizers[role].zero_grad(); loss.backward()
            norm=nn.utils.clip_grad_norm_(actor.parameters(),.5)
            if not torch.isfinite(norm): raise RuntimeError('Non-finite actor gradient')
            optimizers[role].step()
        logs[role]=dict(loss=float(loss.detach()),old_kl=kl,tokens=len(rows),beats=len(old),epochs=epoch+1)
        if auxiliary_loss is not None:logs[role]['auxiliary_loss']=float(auxiliary_loss.detach())
    for _ in range(4):
        loss=.5*(critic(states)-returns).square().mean()
        critic_optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(critic.parameters(),.5); critic_optimizer.step()
    logs['critic_loss']=float(loss.detach())
    return logs

def load_model(device='auto'):
    device=('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device
    dtype=torch.bfloat16 if device=='cuda' and torch.cuda.is_bf16_supported() else torch.float32
    model=MIDIModel.from_pretrained(ROOT/'models/skytnt-tv2o-medium',local_files_only=True,dtype=dtype).to(device).eval()
    model.requires_grad_(False)
    return model

def finish_audio(path,brief):
    """Use the same musical timeline plus two seconds of release for every stem."""
    seconds=brief['bars']*4*60/brief['bpm']+2
    temporary=path.with_name(path.stem+'_trim.wav')
    subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(path),'-af',
                    f'apad,atrim=duration={seconds},afade=t=out:st={seconds-.3}:d=0.3',
                    '-c:a','pcm_s16le',str(temporary)],check=True,capture_output=True)
    temporary.replace(path)
    subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(path),'-codec:a','libmp3lame',
                    '-q:a','2',str(path.with_suffix('.mp3'))],check=True,capture_output=True)

def export_performance(root,name,episode,audio=True):
    from evaluate_baseline import validate_audio
    import mido
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    checks={}; brief=episode['brief']
    # Keep the performance recoverable if the user interrupts an audio render.
    (root/f'{name}_events.json').write_text(json.dumps(episode['notes'],indent=2))
    (root/f'{name}_trace.json').write_text(json.dumps(episode['trace'],indent=2))
    for stem in ['mix',*ROLES]:
        tracks=[[['track_name',0,b'Conductor'],['set_tempo',0,int(60_000_000/brief['bpm'])],['time_signature',0,4,2,24,8]]]
        expected=[]
        for role,s in ROLES.items():
            if stem!='mix' and stem!=role: continue
            notes=[n for n in episode['notes'] if n[3]==s['channel']]; expected.extend(notes)
            tracks.append([['track_name',0,role.encode()],['patch_change',0,s['channel'],s['patch']],*notes])
        path=write_result(root,f'{name}_{stem}',[TPB,*tracks],audio)
        active={}; decoded=[]; tick=0
        for e in mido.merge_tracks(mido.MidiFile(path).tracks):
            tick+=e.time
            if e.type=='note_on' and e.velocity>0:
                key=(e.channel,e.note)
                if key in active: raise AssertionError('Overlapping repeated pitch')
                active[key]=(tick,e.velocity)
            elif e.type=='note_off' or (e.type=='note_on' and e.velocity==0):
                onset,velocity=active.pop((e.channel,e.note)); decoded.append(['note',onset,tick-onset,e.channel,e.note,velocity])
        assert not active and sorted(decoded)==sorted(expected)
        checks[stem]=dict(notes=len(expected),events_preserved=True)
        if audio:
            finish_audio(path.with_suffix('.wav'),brief)
            checks[stem]['audio']=validate_audio(path.with_suffix('.wav'))
    return checks
