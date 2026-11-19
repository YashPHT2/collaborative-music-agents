"""Version 2 offline note policies: learned bar plans and synchronous revisions."""
import copy, contextlib, hashlib, json, math
import torch
from torch import nn
import note_trio as nt
import supervised_ensemble as se

ROLES=list(nt.ROLES);VERSION=2

class Musician(nt.Musician):
    """Neutral residual policy with support for duration token 128."""
    action_size=129
    def __init__(self):
        super().__init__();self.head=nn.Linear(256,self.action_size)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)

@contextlib.contextmanager
def limits():
    old=nt.ROLES['strings']['cap'];nt.ROLES['strings']['cap']=4
    try:yield
    finally:nt.ROLES['strings']['cap']=old

class Sampler(se.EnsembleSampler):
    def constrain_values(self,kind,values,decoded):
        if kind=='duration':
            maximum=min(128,(self.brief.bars*1920-self.beat*480-decoded['time2']*30)//30)
            return list(range(2,maximum+1,2)) or [maximum]
        return super().constrain_values(kind,values,decoded)

def description(notes,bar,total=8):
    ns=[n for n in notes if bar*1920<=n[1]<(bar+1)*1920];v=torch.zeros(32)
    for n in ns:v[n[4]%12]+=n[2]
    v[:12]/=v[:12].sum().clamp_min(1)
    for i,r in enumerate(ROLES):
        own=[n for n in ns if n[3]==i]
        v[12+i]=min(1,len(own)/16)
        if own:
            v[15+i]=sum(n[4] for n in own)/len(own)/127
            v[18+i]=sum(n[5] for n in own)/len(own)/127
            v[21+i]=sum(n[2] for n in own)/len(own)/3840
    earlier={n[4]%12 for n in notes if n[1]<bar*1920}
    v[24]=sum(n[4]%12 not in earlier for n in ns)/max(1,len(ns))
    v[25]=bar/max(1,total-1);v[26]=float(bar%4==0);v[27]=float(bar%4==3)
    return v

class Planner(nn.Module):
    def __init__(self):
        super().__init__();self.gru=nn.GRU(65,128,2,batch_first=True);self.head=nn.Linear(128,32)
    def forward(self,x,hidden=None):
        y,h=self.gru(x,hidden);return self.head(y).sigmoid(),h

def draft_features(notes,role,start):
    if any(n[3]==ROLES.index(role) for n in notes):raise ValueError('Own draft leaked into partner condition')
    rows=[]
    for n in sorted(notes,key=lambda n:(n[1],n[3],n[4]))[:256]:
        if not start<=n[1]<start+3840:raise ValueError('Draft outside announced segment')
        rows.append([(n[1]-start)/3840,n[2]/3840,n[4]/127,n[5]/127,*[float(n[3]==i) for i in range(3)],1.])
    return torch.tensor(rows or [[0.]*8],dtype=torch.float32)

class Conditions(nn.Module):
    def __init__(self):
        super().__init__();self.drafts=nn.GRU(8,128,batch_first=True)
        self.out=nn.Sequential(nn.Linear(163,256),nn.Tanh(),nn.Linear(256,1024))
        nn.init.zeros_(self.out[-1].weight);nn.init.zeros_(self.out[-1].bias)
    def forward(self,plan,drafts,role):
        _,h=self.drafts(drafts.unsqueeze(0));one=torch.eye(3,device=plan.device)[ROLES.index(role)]
        return self.out(torch.cat([plan,h[-1,0],one]))

class ConditionedModel:
    def __init__(self,model,condition):self.model=model;self.condition=condition
    def __getattr__(self,k):return getattr(self.model,k)
    def __call__(self,*a,**kw):return self.model(*a,**kw)
    def forward_token(self,hidden_state=None,x=None,cache=None):
        if hidden_state is not None:hidden_state=hidden_state+self.condition.to(hidden_state.dtype)
        return self.model.forward_token(hidden_state,x,cache)

def candidates(records,bars=8):
    result=[];seen={};priority={'test':0,'validation':1,'train':2}
    for rec in sorted(records,key=lambda r:priority[r['split']]):
        notes=rec['notes'];end=max((n[1]+n[2] for n in notes),default=0)
        for start in range(0,max(0,end-bars*1920+1),bars*1920):
            ns=[]
            for n in notes:
                if start<=n[1]<start+bars*1920:
                    p=copy.deepcopy(n);p[1]-=start;p[2]=min(p[2],bars*1920-p[1])//60*60
                    if p[2]>=60:ns.append(p)
            with limits():ns=se.playable_projection(ns)
            present=[r for i,r in enumerate(ROLES) if sum(n[3]==i for n in ns)>=8]
            if not present:continue
            fp=hashlib.sha256(json.dumps(ns).encode()).hexdigest()
            if fp in seen:continue
            seen[fp]=rec['split'];result.append(dict(song=rec['song'],split=rec['split'],group=rec['group'],notes=ns,
                bars=bars,window_start=start,available_roles=present,fingerprint=fp))
    return result

def compose(model,adapters,agents,planner,conditions,bars=8,seed=9619001,mode='revised',order=None,partner_mode='correct',prefix=None,collect_training=False,plan_transform=None):
    if mode not in ['baseline','planned','revised']:raise ValueError('Unknown composition mode')
    if partner_mode not in ['correct','missing','mismatched']:raise ValueError('Unknown partner mode')
    order=order or ROLES
    if sorted(order)!=sorted(ROLES):raise ValueError('Each role must appear once')
    brief=nt.Brief(bars);ledger=copy.deepcopy(prefix or []);trace=[];decisions=[];plans=[];hidden=None;previous=torch.zeros(32)
    initial_beat=4 if prefix is not None else 0
    if any(n[1]>=initial_beat*480 for n in ledger):raise ValueError('Prefix contains future notes')
    device=next(conditions.parameters()).device;generator=torch.Generator().manual_seed(seed)
    training_records={}
    @torch.no_grad()
    def segment(start,stop,drafts,revision):
        actions={}
        for role in order:
            own=copy.deepcopy(ledger);made=[];adapters.activate(role)
            partner=[n for n in drafts if n[3]!=ROLES.index(role)]
            if partner_mode=='missing':partner=[]
            elif partner_mode=='mismatched':
                partner=copy.deepcopy(partner)
                for n in partner:n[4]=max(0,min(127,n[4]+6))
            features=draft_features(partner,role,start*480).to(device)
            for beat in range(start,stop):
                plan=plans[beat//4].to(device)
                delta=conditions(plan,features,role) if mode!='baseline' else torch.zeros(1024,device=device)
                wrapped=ConditionedModel(model,delta);history=nt.visible(own,beat)
                sampler=Sampler(wrapped,role,seed+beat*1009+ROLES.index(role)*100003+revision*700001,brief,beat,history)
                notes,records=sampler.propose(agents[role],nt.state_features(brief,beat,history))
                if collect_training:training_records[(revision,beat,role)]=records
                action={r:notes if r==role else [] for r in ROLES};own=nt.commit(own,action,beat,brief);made.extend(notes)
            actions[role]=made
        return sorted(sum(actions.values(),[]),key=lambda n:(n[1],n[3],n[4]))
    with limits(),torch.no_grad():
        for start in range(initial_beat,bars*4,8):
            while len(plans)<min(bars,start//4+2):
                bar=len(plans);summary=description(ledger,max(0,start//4-1),bars) if ledger else torch.zeros(32)
                x=torch.cat([previous,summary,torch.tensor([bar/max(1,bars-1)])]).to(device)
                p,hidden=planner(x[None,None],hidden);previous=p[0,0].cpu()
                if plan_transform is not None:
                    previous=plan_transform(previous.clone(),bar,bars,ledger)
                    if previous.shape!=(32,) or not torch.isfinite(previous).all():raise ValueError('Invalid transformed plan')
                plans.append(previous)
            stop=min(start+8,bars*4);draft=segment(start,stop,[],0)
            final=segment(start,stop,draft,1) if mode=='revised' else draft
            for beat in range(start,stop):
                actions={r:[n for n in final if n[3]==i and n[1]//480==beat] for i,r in enumerate(ROLES)}
                ledger=nt.commit(ledger,actions,beat,brief)
                for role,ns in actions.items():decisions.extend(dict(role=role,beat=beat,note=n,origin='revised_policy' if mode=='revised' else 'policy') for n in ns)
            trace.append(dict(start_beat=start,draft=draft,committed=final,mode=mode,partner_mode=partner_mode))
    assert sorted(ledger)==sorted([d['note'] for d in decisions]+list(prefix or []))
    result=dict(notes=ledger,decisions=decisions,trace=trace,plans=[p.tolist() for p in plans],brief=nt.asdict(brief),seed=seed,dataset_prefix=prefix or [])
    if collect_training:
        revision=1 if mode=='revised' else 0
        result['policy_records']={beat:{r:training_records[(revision,beat,r)] for r in ROLES} for beat in range(initial_beat,bars*4)}
    return result

def load(path):
    s=torch.load(path,map_location='cpu',weights_only=False)
    if s.get('format_version')!=VERSION:raise ValueError('Expected offline composition v2 checkpoint')
    if s.get('constraints')!={'strings_cap':4,'max_duration_ticks':3840}:raise ValueError('Unknown offline sampling constraints')
    model=nt.load_model();adapters=se.InstrumentAdapters(model);adapters.load(s['adapters'])
    planner=Planner();planner.load_state_dict(s['planner']);planner.eval()
    conditions=Conditions().to(model.device);conditions.load_state_dict(s['conditions']);conditions.eval()
    agents={r:Musician().eval() for r in ROLES}
    return model,adapters,agents,planner.to(model.device),conditions

@torch.no_grad()
def fixed_response(model,adapters,agent,planner,conditions,w,role,seed,context='correct'):
    """Generate two bars against unchanged real partners after a one-bar prefix."""
    if context not in ['correct','missing','mismatched']:raise ValueError('Unknown context')
    channel=ROLES.index(role);brief=nt.Brief(w['bars']);ledger=nt.visible(w['notes'],4)
    fixed=[copy.deepcopy(n) for n in w['notes'] if n[3]!=channel and 1920<=n[1]<5760]
    device=next(conditions.parameters()).device;hidden=None;previous=torch.zeros(32,device=device);plans={}
    decisions=[];adapters.activate(role)
    with limits():
        for beat in range(4,12):
            bar=beat//4
            if bar not in plans:
                summary=description(ledger,bar-1,w['bars']).to(device)
                x=torch.cat([previous,summary,torch.tensor([bar/(w['bars']-1)],device=device)])
                pred,hidden=planner(x[None,None],hidden);previous=pred[0,0];plans[bar]=previous
            start=beat//8*3840
            partners=[copy.deepcopy(n) for n in w['notes'] if n[3]!=channel and start<=n[1]<start+3840]
            history=nt.visible(ledger,beat)
            if context=='missing':partners=[];history=[n for n in history if n[3]==channel]
            elif context=='mismatched':
                for n in partners:n[4]=min(127,n[4]+6)
                for n in history:
                    if n[3]!=channel:n[4]=min(127,n[4]+6)
            delta=conditions(plans[bar],draft_features(partners,role,start).to(device),role)
            sampler=Sampler(ConditionedModel(model,delta),role,seed+beat*1009,brief,beat,history)
            notes,_=sampler.propose(agent,nt.state_features(brief,beat,history))
            actions={r:[n for n in fixed if n[3]==i and n[1]//480==beat] for i,r in enumerate(ROLES)}
            actions[role]=notes;ledger=nt.commit(ledger,actions,beat,brief);decisions.extend(notes)
    actual=[n for n in ledger if n[3]!=channel and n[1]>=1920]
    assert sorted(actual)==sorted(fixed)
    return ledger,decisions
