"""Resumable eight-hour supervised, plan, and revision campaign. No PPO."""
import argparse, copy, gzip, hashlib, json, random, time
from pathlib import Path
import torch
import torch.nn.functional as F
import offline_ensemble as oe
import composition_data as data
import note_trio as nt
import supervised_ensemble as se

OUT=nt.ROOT/'outputs'/'offline_ensemble_v1'
STAGES=['general','specialize','cinematic','planned','revised']

def transpose(w,rng):
    shifts=[s for s in range(-5,6) if all(n[4]+s in nt.ROLES[oe.ROLES[n[3]]]['pitches'] for n in w['notes'])]
    shift=rng.choice(shifts or [0]);out={**w,'notes':copy.deepcopy(w['notes'])}
    for n in out['notes']:n[4]+=shift
    return out,shift

def cinematic(w):
    """Weak structural selection, not an artist/style annotation."""
    ns=w['notes']
    return len(ns)>=24 and sum(n[2]>=960 for n in ns)/len(ns)>=.25 and len({n[4]%12 for n in ns})>=4

def atomic(path,obj):
    temp=path.with_suffix('.tmp');torch.save(obj,temp);temp.replace(path)

def raw_loss(model,record,index):
    raw=record['raw'];end=max(n['start'] for n in raw);start=(index%max(1,int(end//7680)))*7680
    ns=sorted([n for n in raw if start<=n['start']<start+7680],key=lambda n:(n['start'],n['program'],n['pitch']))[:96]
    if len(ns)<8:return None
    programs=sorted({n['program'] for n in ns});tok=model.tokenizer
    rows=[[tok.bos_id]+[tok.pad_id]*7]
    channels=[i for i in range(16) if i!=9]
    for i,p in enumerate(programs):rows.append(tok.event2tokens(['patch_change',0,0,i+1,channels[i],p]))
    last=0
    for n in ns:
        tick=max(last,round((n['start']-start)/30)*30);i=programs.index(n['program'])
        row=tok.event2tokens(['note',tick//480-last//480,tick%480//30,i+1,channels[i],n['pitch'],max(1,min(127,n['velocity'])),max(2,min(128,round(n['duration']/30)))])
        if row:rows.append(row);last=tick
    x=torch.tensor(rows,device=model.device);hidden=model(x[None,:-1])[0];y=x[1:]
    logits=model.forward_token(hidden,y[:,:-1]).float()
    return F.cross_entropy(logits.reshape(-1,logits.shape[-1]),y.reshape(-1),ignore_index=tok.pad_id)

def plan_loss(planner,w,device):
    target=torch.stack([oe.description(w['notes'],b,w['bars']) for b in range(w['bars'])]).to(device)
    previous=torch.cat([torch.zeros_like(target[:1]),target[:-1]])
    phase=torch.linspace(0,1,w['bars'],device=device)[:,None]
    pred,_=planner(torch.cat([previous,previous,phase],-1)[None])
    return F.mse_loss(pred[0],target)

def teacher_loss(model,conditions,w,role,beat,stage,draft=None):
    ex={**w,'beat':beat,'sampler_type':oe.Sampler}
    wrapped=model
    if stage in ['planned','revised']:
        start=beat//8*3840
        partners=[] if stage=='planned' else [n for n in (w['notes'] if draft is None else draft) if n[3]!=oe.ROLES.index(role) and start<=n[1]<start+3840]
        device=next(conditions.parameters()).device
        delta=conditions(oe.description(w['notes'],beat//4,w['bars']).to(device),oe.draft_features(partners,role,start).to(device),role)
        wrapped=oe.ConditionedModel(model,delta)
    with oe.limits():ex=se.teacher_example(wrapped,role,ex)
    return None if ex is None else se.teacher_loss(wrapped,ex)[0]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--steps',type=int,default=128);p.add_argument('--hours',type=float,default=8)
    p.add_argument('--resume',action='store_true');p.add_argument('--pilot',action='store_true')
    args=p.parse_args();root=args.output;root.mkdir(parents=True,exist_ok=True)
    if args.steps<1 or not 0<args.hours<=8:p.error('Positive steps and hours <=8 required')
    if (root/'progress.pt').exists() and not args.resume:raise ValueError('Use --resume or a new output')
    records=data.build();windows=oe.candidates(records,8)+oe.candidates(records,4)
    train=[w for w in windows if w['split']=='train'];val=[w for w in windows if w['split']=='validation']
    # Bound each composition's contribution; deterministic ordering is resumed exactly.
    def balanced(ws):
        groups={}
        for w in ws:groups.setdefault(w['group'],[]).append(w)
        return [w for k in sorted(groups) for w in groups[k][:16]]
    train=balanced(train);val=balanced(val)
    counts={r:len({w['group'] for w in val if r in w['available_roles'] and len(w['available_roles'])>=2}) for r in oe.ROLES}
    manifest=dict(train_windows=len(train),validation_windows=len(val),validation_compositions=counts,
        gate_passed=all(c>=10 for c in counts.values()),dataset_sha256=hashlib.sha256((data.ROOT/'corpus.json.gz').read_bytes()).hexdigest(),
        windows=[{k:w[k] for k in ['song','split','window_start','bars','fingerprint']} for w in windows])
    (root/'data_manifest.json').write_text(json.dumps(manifest,indent=2))
    if not manifest['gate_passed'] and not args.pilot:
        (root/'REPORT.md').write_text('Data gate failed: fewer than ten independent validation compositions for a role. Use --pilot only for engineering verification; cooperation cannot be established.\n'+json.dumps(counts))
        print('Data gate failed',counts,flush=True);return
    if not train or not val:raise ValueError('Empty training or validation corpus')
    torch.set_num_threads(4);torch.manual_seed(20261019);random.seed(20261019)
    model=nt.load_model();adapters=se.InstrumentAdapters(model);basehash=adapters.base_hash()
    planner=oe.Planner().to(model.device);conditions=oe.Conditions().to(model.device)
    params=[p for r in oe.ROLES for p in adapters.params(r)]+list(planner.parameters())+list(conditions.parameters())
    optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01)
    config=dict(version=2,steps=args.steps,hours=args.hours,pilot=args.pilot,dataset=manifest['dataset_sha256'])
    state=dict(stage=0,step=0,elapsed=0.,logs=[],best=float('inf'),stage_started=False)
    draft_cache={}
    def bundle():return dict(format_version=2,adapters=adapters.state(),planner=planner.state_dict(),conditions=conditions.state_dict(),
        constraints=dict(strings_cap=4,max_duration_ticks=3840),config=config,base_sha256=basehash,roles=nt.ROLES,
        model_revision='0f8f265d4330f4e46527ac2313200254c5757f5f')
    if args.resume:
        saved=torch.load(root/'progress.pt',map_location='cpu',weights_only=False)
        if saved['config']!=config:raise ValueError('Resume recipe differs')
        adapters.load(saved['adapters']);planner.load_state_dict(saved['planner']);conditions.load_state_dict(saved['conditions'])
        optimizer.load_state_dict(saved['optimizer']);state=saved['state'];draft_cache=saved.get('draft_cache',{})
        torch.set_rng_state(saved['rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
    initial_elapsed=state['elapsed'];started=time.monotonic()
    def save():
        state['elapsed']=initial_elapsed+time.monotonic()-started
        atomic(root/'progress.pt',{**bundle(), 'state':state,'optimizer':optimizer.state_dict(),'draft_cache':draft_cache,
            'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()})
        (root/'training_log.json').write_text(json.dumps(state,indent=2))
    def budget():return initial_elapsed+time.monotonic()-started>=args.hours*3600
    @torch.no_grad()
    def validate(stage):
        conditions.eval();planner.eval();scores={}
        for role in oe.ROLES:
            adapters.activate(role);distinct={}
            for w in val:
                if role in w['available_roles']:distinct.setdefault(w['group'],w)
            ws=list(distinct.values())[:12];losses=[]
            for j,w in enumerate(ws):
                loss=teacher_loss(model,conditions,w,role,4+(j*7)%(w['bars']*4-4),stage)
                if loss is not None:losses.append(float(loss))
            scores[role]=sum(losses)/max(1,len(losses))
        return scores
    try:
        while state['stage']<len(STAGES):
            stage=STAGES[state['stage']]
            if not state['stage_started']:
                if stage=='specialize':
                    source=adapters.state();cloned=copy.deepcopy(source)
                    for role in ['strings','cello']:
                        for k in cloned:
                            if f'.{role}.' in k:cloned[k]=source[k.replace(f'.{role}.','.organ.')].clone()
                    adapters.load(cloned)
                optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01)
                state['stage_started']=True;state['start_validation']=validate(stage)
                state['best']=sum(state['start_validation'].values())/3
                atomic(root/f'{stage}.pt',bundle());save()
            for step in range(state['step'],args.steps):
                if budget():save();print('Eight-hour budget reached; resumable checkpoint saved',flush=True);return
                optimizer.zero_grad();rng=random.Random(20261019+state['stage']*100000+step);total=0.;used=0
                for batch in range(8):
                    role=oe.ROLES[(step*8+batch)%3];adapters.activate('organ' if stage=='general' else role,training=True)
                    conditions.train();planner.train()
                    if stage=='general':
                        rec=rng.choice([r for r in records if r['split']=='train']);loss=raw_loss(model,rec,step*8+batch)
                    else:
                        ws=[w for w in train if role in w['available_roles']]
                        focused=[w for w in ws if cinematic(w)]
                        if stage=='cinematic' and focused:ws=focused
                        if stage=='revised' and batch%2==1:ws=sorted(ws,key=lambda w:(w['song'],w['window_start'],w['bars']))[:12]
                        original=rng.choice(ws);w,shift=transpose(original,rng);beat=rng.randrange(w['bars']*4)
                        draft=None
                        if stage=='revised' and batch%2==1:
                            key=w['song']+':'+str(w['window_start'])+':'+str(w['bars'])
                            if key not in draft_cache:
                                # A frozen baseline supplies realistic partner drafts. No validation targets are cached.
                                baseline=torch.load(root/'cinematic.pt',map_location='cpu',weights_only=False)
                                current=adapters.state();adapters.load(baseline['adapters'])
                                agents={r:oe.Musician().eval() for r in oe.ROLES}
                                ep=oe.compose(model,adapters,agents,planner,conditions,w['bars'],step+9619001,'baseline',prefix=nt.visible(original['notes'],4))
                                draft_cache[key]=ep['notes'];adapters.load(current);adapters.activate(role,training=True)
                            draft=copy.deepcopy(draft_cache[key])
                            for n in draft:n[4]=max(0,min(127,n[4]+shift))
                        loss=teacher_loss(model,conditions,w,role,beat,stage,draft)
                        if loss is not None and stage in ['planned','revised']:loss=loss+.1*plan_loss(planner,w,model.device)
                    if loss is None:continue
                    if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                    (loss/8).backward();total+=float(loss.detach());used+=1
                if not used:raise ValueError('No supported teacher decisions')
                if used<8:
                    for par in params:
                        if par.grad is not None:par.grad.mul_(8/used)
                torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step();state['step']=step+1
                print(f'{stage} {step+1}/{args.steps}: loss={total/used:.4f}',flush=True)
                if (step+1)%16==0 or step+1==args.steps:
                    scores=validate(stage);mean=sum(scores.values())/3
                    eligible=stage=='general' or all(scores[r]<=state['start_validation'][r]*1.02 for r in oe.ROLES)
                    state['logs'].append(dict(stage=stage,step=step+1,loss=total/used,validation=scores,eligible=eligible))
                    if eligible and mean<state['best']:
                        state['best']=mean;atomic(root/f'{stage}.pt',bundle())
                    save()
            chosen=torch.load(root/f'{stage}.pt',map_location='cpu',weights_only=False)
            adapters.load(chosen['adapters']);planner.load_state_dict(chosen['planner']);conditions.load_state_dict(chosen['conditions'])
            state.update(stage=state['stage']+1,step=0,stage_started=False);save()
        if adapters.base_hash()!=basehash:raise ValueError('Frozen backbone changed')
        atomic(root/'latest.pt',bundle())
        (root/'REPORT.md').write_text('Supervised campaign complete. Human quality and cooperation gates are PENDING. latest.pt is an endpoint, not a proven quality winner.\n'+json.dumps(state,indent=2))
        print('Campaign complete; listening evaluation required.',flush=True)
    except BaseException:
        # Do not save partially applied gradients as a completed optimizer step.
        raise

if __name__=='__main__':main()
