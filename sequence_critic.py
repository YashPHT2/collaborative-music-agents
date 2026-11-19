"""Causal sequence-aware value critic, not a human musical-quality judge."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import note_trio as nt

TOKENS=256
FEATURES=8
INPUT=32+TOKENS*FEATURES+1


def critic_state(brief,beat,history):
    state=nt.state_features(brief,beat,history)
    start=max(0,(beat-32)*480)
    notes=sorted([n for n in history if n[1]>=start or n[1]+n[2]>start],key=lambda n:(n[1],n[3],n[4]))[-TOKENS:]
    seq=torch.zeros(TOKENS,FEATURES)
    for i,n in enumerate(notes):
        seq[i,:4]=torch.tensor([(n[1]-beat*480)/15360,min(n[2],1920)/1920,n[4]/127,n[5]/127])
        seq[i,4+n[3]]=1;seq[i,7]=1
    return torch.cat([state,seq.flatten(),torch.tensor([float(len(notes))])])


class SequenceCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.event=nn.Sequential(nn.Linear(FEATURES,64),nn.Tanh())
        self.sequence=nn.GRU(64,96,batch_first=True)
        self.value=nn.Sequential(nn.Linear(128,96),nn.Tanh(),nn.Linear(96,1))

    def forward(self,x):
        single=x.ndim==1
        if single:x=x.unsqueeze(0)
        if x.shape[-1]!=INPUT:raise ValueError('Sequence critic requires its versioned sequence input')
        seq=x[:,32:-1].reshape(-1,TOKENS,FEATURES)
        lengths=x[:,-1].long().clamp(1,TOKENS).cpu()
        packed=pack_padded_sequence(self.event(seq),lengths,batch_first=True,enforce_sorted=False)
        _,hidden=self.sequence(packed)
        result=self.value(torch.cat([x[:,:32],hidden[-1]],-1)).squeeze(-1)
        return result.squeeze(0) if single else result
