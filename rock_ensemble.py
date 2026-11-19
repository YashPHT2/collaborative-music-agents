"""Isolated rock role configuration. Never writes orchestral parameters."""
import os
os.environ.setdefault('USE_TF','0');os.environ.setdefault('USE_FLAX','0')
import copy
import torch
import orchestra_learning as l
from band import BandSampler

SPECS=[('guitar_1','guitar',29,0,8),('guitar_2','guitar',30,1,8),
       ('bass','bass',33,2,4),('drums','drums',0,9,16),('keyboard','keyboard',0,3,8)]
ROLES={name:dict(id=i,family=f,patch=p,channel=c,track=i+1,pitches=list(range(128)),
    cap=cap,polyphony=cap,grid=list(range(16)),entry=0) for i,(name,f,p,c,cap) in enumerate(SPECS)}
FAMILIES=list(dict.fromkeys(s['family'] for s in ROLES.values()))

class Sampler(BandSampler):
    def __init__(self,model,role,seed,brief,beat,history):
        local=[]
        for n in history:
            n=copy.deepcopy(n);n[3]=list(ROLES.values())[n[3]]['channel'];local.append(n)
        super().__init__(model,role,seed,brief,beat,local,roles=ROLES)
        self.rows=[r for r in self.rows if r[0]!=self.tok.event_ids['key_signature']]
    def onsets(self):
        return [s for s in range(16) if self.beat*480+s*30>=self.last_tick and
            self.brief.bars*1920-(self.beat*480+s*30)>=60 and
            sum(n[3]==self.settings['channel'] and n[1]<=self.beat*480+s*30<n[1]+n[2] for n in self.observed)<self.settings['polyphony']]
    def constrain_values(self,kind,values,decoded):
        if kind=='velocity':return list(range(32,125,4))
        if kind=='duration':
            maximum=min(128,(self.brief.bars*1920-self.beat*480-decoded['time2']*30)//30)
            return [2] if self.role=='drums' else list(range(2,maximum+1,2))
        return values

def configure():
    # These globals are confined to this rock process, not persisted in source.
    l.ROLES=ROLES;l.FAMILIES=FAMILIES;l.Sampler=Sampler

def teacher(model,w,role,beat,history_override=None,allow_warmup=False):
    """Past-only joint context, private current-beat targets, explicit stop labels."""
    import note_trio as nt
    import supervised_ensemble as se
    if role not in w['available_roles'] or not (0 if allow_warmup else 8)<=beat<24:return None
    history=l.visible(w['notes'],beat) if history_override is None else copy.deepcopy(history_override)
    if any(n[1]>=beat*480 for n in history):raise ValueError('Future note in teacher history')
    sampler=Sampler(model,role,0,nt.Brief(6,w['bpm'],0,False),beat,history)
    target=sorted([n for n in w['notes'] if n[3]==ROLES[role]['id'] and n[1]//480==beat],key=lambda n:(n[1],n[4]))
    labels=[];supports=[];positions=[]
    for index in range(len(target)+1):
        if index==len(target) and len(target)==sampler.settings['cap']:break
        if not sampler.onsets():
            if index<len(target):raise ValueError('Target under full polyphony')
            break
        if index<len(target):
            n=target[index];values=dict(event=0,time1=beat-sampler.last_tick//480,time2=n[1]%480//30,
                track=sampler.settings['track'],channel=sampler.settings['channel'],pitch=n[4],velocity=n[5],duration=n[2]//30)
        else:values=dict(event=1)
        row=[sampler.tok.pad_id]*8;mask=torch.zeros(8,sampler.tok.vocab_size,dtype=torch.bool);decoded={}
        for j,key in enumerate(se.KINDS):
            legal,ids=sampler.legal_parameters(key,decoded)
            if values[key] not in legal:raise ValueError(f'Unsupported {role}/{key}/{values[key]}')
            row[j]=ids[legal.index(values[key])]
            if len(ids)>1:mask[j,ids]=True
            decoded[key]=values[key]
            if j==0 and values[key]==1:break
        positions.append(len(sampler.rows)-1);labels.append(row);supports.append(mask)
        if index<len(target):
            n=copy.deepcopy(target[index]);n[3]=sampler.settings['channel'];sampler.observed.append(n);sampler.append_note(n,sampler.settings['track'])
    if not labels:return None
    kind='onset' if target else 'sustain' if any(n[3]==ROLES[role]['id'] and n[1]+n[2]>beat*480 for n in history) else 'silence'
    return dict(x=torch.tensor(sampler.rows),positions=torch.tensor(positions),labels=torch.tensor(labels),support=torch.stack(supports),kind=kind,beat=beat,role=role)
