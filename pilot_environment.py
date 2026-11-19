"""PPO pilot sampler, fresh scripted melodies, and an explicitly heuristic reward."""
import math
import random

from accompany import EventSampler, TPB, resolve_overlaps
from ppo_adapter import ACTIONS, features
import torch


def synthetic_episode(seed, bars=4, condition=None):
    """Different seeds for training/validation/final test; no old ballot themes."""
    rng = random.Random(seed)
    tonic = rng.choice([0,2,5,7])
    scale = [0,2,4,5,7,9,11]
    position = rng.randrange(2,5)
    lead, harmony = [], []
    changed = condition or rng.choice(['pitch','rhythm','both','none'])
    for bar in range(bars):
        key = (tonic + (2 if bar>=bars//2 and changed in ['pitch','both'] else 0))%12
        triad = rng.choice([[0,4,7],[5,9,0],[7,11,2],[9,0,4]])
        harmony.append(dict(start=bar*1920,end=(bar+1)*1920,
                            chord=sorted((key+pc)%12 for pc in triad),
                            scale=sorted((key+pc)%12 for pc in scale)))
        rhythm = [0,360,720,1320] if bar>=bars//2 and changed in ['rhythm','both'] else [0,480,960,1440]
        for index,offset in enumerate(rhythm):
            position = max(0,min(7,position+rng.choice([-2,-1,0,1,2])))
            pitch = 72+key+scale[position%7]+12*(position//7)
            next_offset = rhythm[index+1] if index<3 else 1920
            duration = max(30,((next_offset-offset)*3//4)//30*30)
            lead.append(['note',bar*1920+offset,duration,0,pitch,90])
    return lead,harmony,changed


def window_reward(lead, all_notes, start, end):
    """Local support proxy, not a learned human preference or music-quality score.

    Stronger penalties for semitone/tritone tension are provisional. They cannot
    distinguish expressive dissonance from mistakes; KL limits the pilot's drift.
    No future melody pitch, annotated key, or chord is used here.
    """
    active = [n for n in all_notes if n[1]<end and n[1]+n[2]>start]
    attacks = [n for n in active if n[1]>=start]
    if not active:
        return -0.6
    total, support = 0,0
    for note in active:
        weight = min(end,note[1]+note[2])-max(start,note[1])
        interval = abs(lead[4]-note[4])%12
        compatibility = 1 if interval in [0,3,4,5,7,8,9] else (-1 if interval in [1,6,11] else 0)
        total += weight
        support += weight*compatibility
    boundaries = sorted({start,end}|{max(start,n[1]) for n in active}|{min(end,n[1]+n[2]) for n in active})
    coverage = sum(b-a for a,b in zip(boundaries,boundaries[1:])
                   if any(n[1]<b and n[1]+n[2]>a for n in active))/(end-start)
    density = len(attacks)/((end-start)/TPB)
    history = all_notes[-16:]
    dominant = max((sum(n[4]==p for n in history)/len(history) for p in {n[4] for n in history}),default=0)
    return 0.5*coverage+0.4*support/total-0.25*max(0,(dominant-0.5)/0.5)-0.2*min(3,max(0,(density-3)/3))


class AdapterSampler(EventSampler):
    def __init__(self, model, policy, seed, bpm, temperature, end_tick, train=False):
        super().__init__(model,seed,bpm,temperature)
        self.policy = policy  # Tiny FP32 CPU policy; base model remains on GPU.
        self.end_tick = end_tick
        self.train_rollout = train
        self.trajectory = []
        self.history = []
        self.current_melody = 72
        self.window_notes = 0
        self.pending = []

    def append_note(self, note, track):
        super().append_note(note,track)
        if track == 1:
            self.current_melody = note[4]
            self.window_notes = 0
        else:
            self.history.append(note[4])
            self.window_notes += 1

    def choose_token(self,index,parameters,logits,probs,allowed,row):
        kind = 'event' if index==0 else parameters[index-1]
        if kind not in ['event','pitch']:
            return super().choose_token(index,parameters,logits,probs,allowed,row)
        reference = torch.zeros(ACTIONS)
        if kind == 'event':
            token_ids = [self.tok.event_ids['note'],self.tok.eos_id]
            base = torch.softmax(logits[0,token_ids].float()/self.temperature,dim=-1).cpu()
            # Fixed exploration floor gives WAIT a nonzero probability even if
            # the pretrained model assigns it negligible mass. Matched baseline
            # uses the same floor, so this is not credited to PPO.
            reference[:2] = 0.95*base+0.05/2
            action_ids = list(range(2))
        else:
            token_ids = self.tok.parameter_ids['pitch'][36:67]
            base = torch.softmax(logits[0,token_ids].float()/self.temperature,dim=-1)
            ordered,indices = base.sort(descending=True)
            keep = (ordered.cumsum(0)-ordered<=0.98) & (torch.arange(len(ordered),device=base.device)<20)
            filtered = torch.zeros_like(base)
            filtered[indices[keep]] = base[indices[keep]]
            reference[2:] = (filtered/filtered.sum()).cpu()
            action_ids = list(range(2,33))
        observation = features(reference,self.current_melody,self.history[-1] if self.history else None,
                               self.window_notes,self.last_tick,self.end_tick,kind=='pitch',self.history)
        distribution,value = self.policy.distribution(observation.unsqueeze(0),reference.unsqueeze(0))
        sample = torch.multinomial(distribution.probs.to(self.model.device),1,generator=self.generator).item()
        token = token_ids[action_ids.index(sample)]
        if self.train_rollout:
            self.pending.append(dict(observation=observation.clone(),reference=reference.clone(),action=sample,
                                     logp=distribution.log_prob(torch.tensor([sample])).item(),value=value.item(),
                                     kind=kind,reward=0.0))
        return torch.tensor([[token]],device=self.model.device,dtype=torch.long)


@torch.no_grad()
def rollout(model,policy,lead,seed,train=False):
    end_tick = math.ceil((lead[-1][1]+lead[-1][2])/1920)*1920
    sampler = AdapterSampler(model,policy,seed,100,0.9,end_tick,train)
    notes, rewards = [],[]
    stats = dict(eos_waits=0,boundary_waits=0,density_cap_waits=0)
    for index,note in enumerate(lead):
        start = note[1]
        boundary = lead[index+1][1] if index+1<len(lead) else end_tick
        sampler.append_note(note,1)
        window_decisions = []
        for _ in range(16):
            sampler.pending = []
            candidate = sampler.sample()
            if candidate is None:
                window_decisions.extend(sampler.pending)
                stats['eos_waits'] += 1
                break
            if candidate[1]>=boundary:
                # Pitch of a discarded candidate cannot affect the music.
                window_decisions.extend(s for s in sampler.pending if s['kind']=='event')
                stats['boundary_waits'] += 1
                break
            window_decisions.extend(sampler.pending)
            candidate[2] = min(candidate[2],end_tick-candidate[1])
            sampler.append_note(candidate,2)
            notes.append(candidate)
        else:
            stats['density_cap_waits'] += 1
        reward = window_reward(note,notes,start,boundary)
        rewards.append(reward)
        # One environmental window reward, delivered at its final decision.
        # Do not multiply a window reward by the number of generated tokens.
        if window_decisions:
            window_decisions[-1]['reward'] = reward
            sampler.trajectory.extend(window_decisions)
    return resolve_overlaps(notes),sampler.trajectory,rewards,stats
