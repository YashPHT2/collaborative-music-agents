"""Trainable orchestral agents: seven shared family adapters, 19 role conditions."""
import copy
from collections import Counter
import hashlib
import json
import math

import numpy as np
import torch
from torch import nn
from peft import LoraConfig,LoraModel

import generate_full_orchestra as source
import note_trio as nt
import offline_ensemble as oe
import supervised_ensemble as se
import ensemble_repair as er

ROLES=copy.deepcopy(source.ROLES)
FAMILIES=list(source.FAMILIES)
CHANNELS=dict(source.CHANNELS)
SIZE=128
for role,s in ROLES.items():
    s['pitches']=list(range(min(s['pitches']),max(s['pitches'])+1));s['entry']=0;s['grid']=list(range(16))
    s['cap']=4 if s['family']=='pianos' else 3 if s['family'] in ('organ','synth','choir') else 2
    if s['family']=='pianos':s['polyphony']=4
    if s['family'] in ('synth','choir'):s['polyphony']=3


class Sampler(source.OrchestraSampler):
    def __init__(self,model,role,seed,brief,beat,history):
        super().__init__(model,role,seed,brief,beat,history,ROLES,CHANNELS)
        self.rows=[row for row in self.rows if row[0]!=self.tok.event_ids['key_signature']]
        self.exploration=.02
    def onsets(self):
        # Durations use a 60-tick grid. A last onset 30 ticks from the end
        # otherwise reaches an empty duration mask during sampled rollouts.
        return [step for step in super().onsets()
                if self.brief.bars*1920-(self.beat*480+step*30)>=60]
    def constrain_values(self,kind,values,decoded):
        calm=getattr(self.brief,'calm',False)
        if kind=='velocity':
            maximum=min(84,max(32,int(getattr(self.brief,'velocity_max',52 if calm else 84))))
            return list(range(32,maximum+1,4))
        if kind=='duration':
            maximum=min(128,(self.brief.bars*1920-self.beat*480-decoded['time2']*30)//30)
            return list(range(2,maximum+1,2))
        if kind=='pitch' and calm:return [p for p in values if p%12 in source.SCALE]
        return values


def commit(ledger,actions,beat,bars):
    if set(actions)!=set(ROLES):raise ValueError('All roles must submit together')
    additions=[]
    for role,s in ROLES.items():
        ns=actions[role]
        if len(ns)>s['cap']:raise ValueError('Role event cap')
        for n in ns:
            if not (n[0]=='note' and n[3]==s['id'] and beat*480<=n[1]<(beat+1)*480 and n[1]%30==0
                    and n[2]>0 and n[2]%60==0 and n[2]<=3840 and n[1]+n[2]<=bars*1920
                    and n[4] in s['pitches'] and n[5] in range(32,85,4)):raise ValueError('Unsupported orchestral note')
        own=[n for n in ledger+ns if n[3]==s['id']]
        for n in ns:
            active=[p for p in own if p[1]<=n[1]<p[1]+p[2]]
            if len(active)>s['polyphony'] or len({p[4] for p in active})!=len(active):raise ValueError('Voice overlap')
        additions.extend(ns)
    return sorted(ledger+additions,key=lambda n:(n[1],n[3],n[4]))


def visible(notes,beat):return [copy.deepcopy(n) for n in notes if n[1]<beat*480 and (n[1]>=(beat-24)*480 or n[1]+n[2]>beat*480)]


class Adapters:
    def __init__(self,model):
        self.model=model;configs={f:LoraConfig(r=8,lora_alpha=16,lora_dropout=0.,target_modules=['q_proj','v_proj']) for f in FAMILIES}
        self.wrapper=LoraModel(model,configs,adapter_name=FAMILIES[0])
        for family in FAMILIES[1:]:self.wrapper.inject_adapter(model,family)
        model.net.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        self.activate(FAMILIES[0])
    def activate(self,family,training=False):
        self.wrapper.set_adapter(family);self.model.requires_grad_(False);self.model.train(training)
        if training:
            for p in self.params(family):p.requires_grad_(True)
    def params(self,family):return [p for n,p in self.model.named_parameters() if 'lora_' in n and f'.{family}.' in n]
    def state(self):return {n:p.detach().cpu().clone() for n,p in self.model.state_dict().items() if 'lora_' in n}
    def load(self,weights):self.model.load_state_dict(weights,strict=False)
    def base_hash(self):
        h=hashlib.sha256()
        for n,p in self.model.state_dict().items():
            if 'lora_' not in n:h.update(n.encode());h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()


def description(notes,bar,bars,available=None):
    v=torch.zeros(SIZE);left=bar*1920;right=left+1920
    current=[n for n in notes if left<=n[1]<right]
    for n in current:v[n[4]%12]+=n[2];v[88+(n[1]%1920)//120]+=1
    v[:12]/=v[:12].sum().clamp_min(1);v[88:104]/=v[88:104].sum().clamp_min(1)
    for role,s in ROLES.items():
        i=s['id'];own=[n for n in current if n[3]==i]
        v[12+i]=min(1,len(own)/16)
        if own:v[31+i]=np.mean([n[4] for n in own])/127;v[50+i]=np.mean([n[5] for n in own])/127
        v[69+i]=sum(any(n[3]==i and n[1]<=t<n[1]+n[2] for n in notes) for t in range(left,right,120))/16
        v[104+i]=float(role in available) if available is not None else float(any(n[3]==i for n in notes))
    v[123]=bar/max(1,bars-1)
    return v


def style(notes,available):
    own=[n for n in notes if n[1]>=3840];v=torch.zeros(4)
    if not own:return v
    v[0]=min(1,len(own)/max(1,len(available)*64));v[1]=np.mean([n[5] for n in own])/127
    v[2]=np.mean([n[2] for n in own])/3840
    chroma=np.zeros(12)
    for n in own:chroma[n[4]%12]+=n[2]
    major=np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
    minor=np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
    correlations=[np.corrcoef(chroma,np.roll(template,k))[0,1] if np.std(chroma)>0 else 0 for template in (major,minor) for k in range(12)]
    v[3]=float(max(correlations[:12])>max(correlations[12:]))
    return v


class Planner(nn.Module):
    def __init__(self):
        super().__init__();self.gru=nn.GRU(SIZE*2+5,128,2,batch_first=True);self.out=nn.Linear(128,SIZE)
    def forward(self,x,hidden=None):
        y,hidden=self.gru(x,hidden);return self.out(y).sigmoid(),hidden


def plan_sequence(planner,notes,bars,available,goal,calm=None):
    device=next(planner.parameters()).device;previous=torch.zeros(SIZE,device=device);hidden=None;plans=[]
    for start in range(0,bars*4,8):
        summary=description(visible(notes,start),start//4-1,bars,available).to(device) if start else previous.new_zeros(SIZE)
        for bar in range(start//4,min(bars,start//4+2)):
            x=torch.cat([previous,summary,goal.to(device),previous.new_tensor([bar/max(1,bars-1)])])
            pred,hidden=planner(x[None,None],hidden);previous=pred[0,0]
            if calm is not None:
                previous=previous.clone()
                previous[12:31]=torch.minimum(previous[12:31],previous.new_tensor(calm['density']))
                previous[50:69]=previous.new_tensor(calm['velocity'])
            plans.append(previous)
    return torch.stack(plans)


class Conditions(nn.Module):
    def __init__(self):
        super().__init__();self.roles=nn.Embedding(len(ROLES),32);self.drafts=nn.GRU(8,64,batch_first=True)
        self.out=nn.Sequential(nn.Linear(SIZE+32+64+4,256),nn.Tanh(),nn.Linear(256,1024))
        nn.init.zeros_(self.out[-1].weight);nn.init.zeros_(self.out[-1].bias)
    def forward(self,plan,draft,role,goal):
        _,hidden=self.drafts(draft[None].to(plan.device));embedding=self.roles(torch.tensor(ROLES[role]['id'],device=plan.device))
        return self.out(torch.cat([plan,embedding,hidden[-1,0],goal.to(plan.device)]))


def draft_features(notes,role,start):
    if any(n[3]==ROLES[role]['id'] for n in notes):raise ValueError('Own target leaked into partner draft')
    rows=[]
    for n in sorted(notes,key=lambda n:(n[1],n[3],n[4]))[:256]:
        if not start*480<=n[1]<(start+8)*480:continue
        s=list(ROLES.values())[n[3]]
        rows.append([(n[1]-start*480)/3840,n[2]/3840,n[4]/127,n[5]/127,n[3]/18,FAMILIES.index(s['family'])/6,1.,0.])
    return torch.tensor(rows or [[0.]*8])


def kind(w,role,beat):
    own=[n for n in w['notes'] if n[3]==ROLES[role]['id']]
    if any(beat*480<=n[1]<(beat+1)*480 for n in own):return 'onset'
    return 'sustain' if any(n[1]<(beat+1)*480 and n[1]+n[2]>beat*480 for n in own) else 'silence'


def teacher(model,w,role,beat):
    if role not in w['available_roles'] or not 8<=beat<24:return None
    history=visible(w['notes'],beat);sampler=Sampler(model,role,0,nt.Brief(6,70,0,False),beat,history)
    target=sorted([copy.deepcopy(n) for n in w['notes'] if n[3]==ROLES[role]['id'] and n[1]//480==beat],key=lambda n:(n[1],n[4]))
    labels=[];supports=[];positions=[]
    if not sampler.onsets():
        if target:raise ValueError('Teacher note under forced rest')
        return None
    for index in range(len(target)+1):
        if index==len(target) and len(target)==sampler.settings['cap']:break
        if not sampler.onsets():
            if index<len(target):raise ValueError('Teacher target unsupported')
            break
        if index<len(target):
            n=target[index];values=dict(event=0,time1=beat-sampler.last_tick//480,time2=n[1]%480//30,
                track=sampler.settings['track'],channel=sampler.settings['channel'],pitch=n[4],velocity=n[5],duration=n[2]//30)
        else:values=dict(event=1)
        row=[sampler.tok.pad_id]*8;mask=torch.zeros(8,sampler.tok.vocab_size,dtype=torch.bool);decoded={}
        for j,key in enumerate(se.KINDS):
            legal,ids=sampler.legal_parameters(key,decoded)
            if values[key] not in legal:raise ValueError(f'Teacher outside support {role}/{key}/{values[key]}')
            row[j]=ids[legal.index(values[key])]
            if len(ids)>1:mask[j,ids]=True
            decoded[key]=values[key]
            if j==0 and values[key]==1:break
        positions.append(len(sampler.rows)-1);labels.append(row);supports.append(mask)
        if index<len(target):
            local=copy.deepcopy(target[index]);local[3]=sampler.settings['channel']
            sampler.observed.append(local);sampler.append_note(local,sampler.settings['track'])
    if not labels:return None
    return dict(x=torch.tensor(sampler.rows),positions=torch.tensor(positions),labels=torch.tensor(labels),support=torch.stack(supports),
                kind=kind(w,role,beat),beat=beat,role=role)


def category_weights(examples):
    from supported_ensemble_loader import supported_categories
    counts={k:Counter() for k in er.CATEGORIES}
    for ex in examples:
        for key in supported_categories(ex):counts[key][ex['kind']=='onset']+=1
    available=sum(bool(v) for v in counts.values())
    return [{k:1/(available*len(counts[k])*counts[k][ex['kind']=='onset']) for k in supported_categories(ex)} for ex in examples]


@torch.no_grad()
def private_draft(model,adapters,agents,planner,conditions,ledger,start,bars,available,goal,seed,calm=None,bpm=70):
    plans=plan_sequence(planner,ledger,bars,available,goal,calm);draft=[]
    brief=nt.Brief(bars,bpm,0,False);brief.calm=calm is not None
    if calm is not None and 'velocity_max' in calm:brief.velocity_max=calm['velocity_max']
    for role in available:
        adapters.activate(ROLES[role]['family']);own=copy.deepcopy(ledger)
        for beat in range(start,min(start+8,bars*4)):
            delta=conditions(plans[beat//4],torch.zeros(1,8,device=model.device),role,goal)
            sampler=Sampler(oe.ConditionedModel(model,delta),role,seed+beat*1009+ROLES[role]['id']*100003,brief,beat,visible(own,beat))
            made,_=sampler.propose(agents[role],torch.zeros(32))
            for n in made:n[3]=ROLES[role]['id']
            own=commit(own,{r:made if r==role else [] for r in ROLES},beat,bars);draft.extend(made)
    return sorted(draft,key=lambda n:(n[1],n[3],n[4]))


@torch.no_grad()
def compose(model,adapters,agents,planner,conditions,bars=8,seed=9919101,mode='planned',calm=None,order=None,progress=None,bpm=70):
    available=list(ROLES);goal=torch.tensor(calm['style'] if calm else [.25,.5,.3,.5],device=model.device)
    ledger=[];decisions=[];trace=[];brief=nt.Brief(bars,bpm,0,False);brief.calm=calm is not None
    if calm is not None and 'velocity_max' in calm:brief.velocity_max=calm['velocity_max']
    for start in range(0,bars*4,8):
        plans=plan_sequence(planner,ledger,bars,available,goal,calm)
        drafts=private_draft(model,adapters,agents,planner,conditions,ledger,start,bars,available,goal,seed,calm,bpm=bpm) if mode=='revised' else []
        for beat in range(start,min(start+8,bars*4)):
            past=visible(ledger,beat);actions={}
            for role in order or ROLES:
                adapters.activate(ROLES[role]['family'])
                partner=[n for n in drafts if n[3]!=ROLES[role]['id']]
                delta=conditions(plans[beat//4],draft_features(partner,role,start).to(model.device),role,goal)
                sampler=Sampler(oe.ConditionedModel(model,delta),role,seed+beat*1009+ROLES[role]['id']*100003,brief,beat,past)
                made,_=sampler.propose(agents[role],torch.zeros(32))
                for n in made:n[3]=ROLES[role]['id']
                actions[role]=made;decisions.extend(dict(role=role,beat=beat,note=n) for n in made)
            ledger=commit(ledger,actions,beat,bars)
            trace.append(dict(beat=beat,emitted={r:len(ns) for r,ns in actions.items()}))
        if progress is not None:progress(min(bars,start//4+2),len(ledger))
    assert sorted(ledger)==sorted(d['note'] for d in decisions)
    return dict(notes=ledger,decisions=decisions,trace=trace,brief=nt.asdict(brief),seed=seed,mode=mode)
