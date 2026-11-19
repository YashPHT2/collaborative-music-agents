"""Bounded supervised rock adapter pilot; not mood-labelled or RL training."""
import os
os.environ.setdefault('USE_TF','0');os.environ.setdefault('USE_FLAX','0')
import collections,copy,gzip,hashlib,json,math,random,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import rock_ensemble as rock
import orchestra_learning as l
import ensemble_repair as er
import note_trio as nt

ROOT=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');torch.save(value,tmp);tmp.replace(path)
def write(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
def chunks(items,capacity):
    chunk=[];maximum=0
    for row in sorted(items,key=lambda row:len(row[0]['x'])):
        length=len(row[0]['x'])
        if chunk and (len(chunk)>=capacity or max(maximum,length)*(len(chunk)+1)>32768):yield chunk;chunk=[];maximum=0
        chunk.append(row);maximum=max(maximum,length)
    if chunk:yield chunk

def run(output,steps=128,hours=2,on_save=lambda:None,smoke=False):
    rock.configure();out=Path(output);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(8);torch.manual_seed(9976001);random.seed(9976001);np.random.seed(9976001)
    data=ROOT/'outputs/rock_training_v1';manifest=json.loads((data/'rock_manifest.json').read_text())
    if not manifest['pilot_data_gate_passed']:raise ValueError('Rock composition coverage gate failed')
    assert sha(data/'rock_examples.json.gz')==manifest['examples_sha256']
    with gzip.open(data/'rock_examples.json.gz','rt') as f:windows=json.load(f)
    if (out/'progress.pt').exists():raise ValueError('Existing run preserved; choose a new output directory')
    init=ROOT/'outputs/preferred_model_v1/step224.pt';init_sha=sha(init)
    saved=torch.load(init,map_location='cpu',weights_only=False)
    model=nt.load_model();adapters=l.Adapters(model);base_hash=adapters.base_hash()
    assert base_hash==saved['base_sha256']
    for family in rock.FAMILIES:
        for p in adapters.params(family):p.data=p.data.float()
    planner=l.Planner().to(model.device);planner.load_state_dict(saved['planner'])
    conditions=l.Conditions().to(model.device) # separate fresh rock conditions
    params=[p for family in rock.FAMILIES for p in adapters.params(family)]+list(conditions.parameters())
    optimizer=torch.optim.AdamW([dict(params=params,lr=5e-5),dict(params=planner.parameters(),lr=1e-4)],weight_decay=.01)
    groups={s:{r:collections.defaultdict(list) for r in rock.ROLES} for s in ['train','validation','test']}
    for w in windows:
        for role in w['available_roles']:groups[w['split']][role][w['group']].append(w)
    partitions={s:{w['group'] for w in windows if w['split']==s} for s in groups}
    assert all(not partitions[a]&partitions[b] for a,b in [('train','validation'),('train','test'),('validation','test')])
    fixed={r:[min(groups['validation'][r][g],key=lambda w:w['id']) for g in sorted(groups['validation'][r])[:3]] for r in rock.ROLES}
    cache=collections.OrderedDict();started=time.monotonic();allparams=params+list(planner.parameters())
    state=dict(stage='rock_supervised_phrase_pilot',step=0,total=steps,complete=False,selected_step=0,updates=[],validations=[],happy_mood_trained=False,rl_trained=False)
    config=dict(steps=steps,hours=hours,phrases_per_role=8,physical_capacity=64,padded_token_budget=32768,adapter_lr=5e-5,planner_lr=1e-4,rank=8,alpha=16,init_sha256=init_sha,data_sha256=manifest['examples_sha256'],seed=9976001)
    def teachers(w,role):
        key=(w['id'],role)
        if key not in cache:
            examples=[rock.teacher(model,w,role,b) for b in range(8,24)];examples=[x for x in examples if x is not None]
            if not examples:raise ValueError('Empty supervised phrase')
            cache[key]=examples
            if len(cache)>64:cache.popitem(last=False)
        cache.move_to_end(key);return cache[key]
    def bundle():
        return dict(format_version='rock-pilot-v1',roles=rock.ROLES,families=rock.FAMILIES,adapters=adapters.state(),planner={k:v.detach().cpu() for k,v in planner.state_dict().items()},conditions={k:v.detach().cpu() for k,v in conditions.state_dict().items()},base_sha256=base_hash,config=config,state=copy.deepcopy(state),optimizer=optimizer.state_dict(),rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),python=random.getstate(),numpy=np.random.get_state()))
    def persist():
        state['elapsed_seconds']=time.monotonic()-started
        atomic(out/'progress.pt',bundle());write(out/'training_log.json',state);on_save()
    @torch.no_grad()
    def validate():
        scores={};planner.eval();conditions.eval()
        for role,s in rock.ROLES.items():
            adapters.activate(s['family']);rows=[]
            for w in fixed[role]:
                goal=torch.tensor(w['style'],device=model.device);plans=l.plan_sequence(planner,w['notes'],6,w['available_roles'],goal)
                examples=teachers(w,role)
                for chunk in chunks([(ex,None) for ex in examples],32):
                    exs=[row[0] for row in chunk]
                    deltas=[conditions(plans[ex['beat']//4],torch.zeros(1,8,device=model.device),role,goal) for ex in exs]
                    losses=er.batch_category_losses(model,exs,deltas)
                    rows.extend(dict(group=w['group'],kind=ex['kind'],losses={k:float(v) for k,v in value.items()}) for ex,value in zip(exs,losses))
            scores[role]=er.aggregate(rows)
            if scores[role]['attributes'] is None:raise ValueError('No validation note attributes for '+role)
            print(json.dumps(dict(event='validation_role',role=role,event_loss=scores[role]['event'],attribute_loss=scores[role]['attributes'])),flush=True)
        return scores
    def score(scores):return sum(scores[r]['event'] for r in rock.ROLES)/len(rock.ROLES)
    def eligible(scores,baseline):
        return score(scores)<score(baseline) and all(scores[r]['attributes']<=baseline[r]['attributes']*1.02 and scores[r]['event']<=baseline[r]['event']*1.02 for r in rock.ROLES)
    # Build all labels before any optimizer update; fail on unsupported targets.
    for w in windows:
        w['style']=l.style([n for n in w['notes'] if n[3]!=rock.ROLES['drums']['id']],w['available_roles']).tolist()
        for role in w['available_roles']:
            if w['split']=='test':continue
            teachers(w,role)
    print(json.dumps(dict(event='label_preflight_passed',windows=len(windows),gpu=torch.cuda.get_device_name() if model.device.type=='cuda' else 'CPU')),flush=True)
    # Measure actual forwards/backwards, without stepping the optimizer.
    # Statistical batch remains eight phrases per role regardless of microbatch.
    if model.device.type!='cuda':raise RuntimeError('Pilot training requires CUDA')
    calibration=[]
    for candidate in [16,32,64]:
        try:
            optimizer.zero_grad(set_to_none=True);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            for role,s in rock.ROLES.items():
                adapters.activate(s['family'],training=True)
                pool=[w for w in windows if w['split']=='train' and role in w['available_roles']]
                longest=max(pool,key=lambda w:len(w['notes']));exs=teachers(longest,role)
                items=[(exs[i%len(exs)],None) for i in range(candidate)]
                for chunk in chunks(items,candidate):
                    values=er.batch_category_losses(model,[r[0] for r in chunk],[torch.zeros(1024,device=model.device) for _ in chunk],training=True)
                    loss=sum(sum(v.values())/len(v) for v in values)/len(values);loss.backward()
                    optimizer.zero_grad(set_to_none=True)
            peak=torch.cuda.max_memory_reserved();calibration.append(dict(capacity=candidate,peak_reserved_bytes=peak))
            if peak>.85*torch.cuda.get_device_properties(0).total_memory:break
        except torch.cuda.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True);torch.cuda.empty_cache();break
    if not calibration:raise RuntimeError('Microbatch calibration failed before any optimizer update')
    capacity=calibration[-1]['capacity'];config['physical_capacity']=capacity;write(out/'batch_calibration.json',calibration)
    baseline=validate();state['initial_validation']=baseline;best=score(baseline)
    atomic(out/'initial.pt',bundle());atomic(out/'selected.pt',bundle());persist()
    for step in range(steps):
        if time.monotonic()-started>=hours*3600:break
        rng=random.Random(9976001+step);tick=time.monotonic();optimizer.zero_grad(set_to_none=True)
        planner.train();conditions.train();total=0.;count=0;role_counts={};batches=[]
        for family in rock.FAMILIES:
            adapters.activate(family,training=True);items=[];roles=[r for r,s in rock.ROLES.items() if s['family']==family]
            for role in roles:
                role_counts[role]=0
                for phrase in range(1 if smoke else 8):
                    group=rng.choice(sorted(groups['train'][role]));w=rng.choice(groups['train'][role][group]);exs=teachers(w,role)
                    goal=torch.tensor(w['style'],device=model.device);plans=l.plan_sequence(planner,w['notes'],6,w['available_roles'],goal)
                    target=torch.stack([l.description(w['notes'],b,6,w['available_roles']) for b in range(6)]).to(model.device)
                    norm=len(rock.FAMILIES)*len(roles)*(1 if smoke else 8)
                    ploss=.1*F.mse_loss(plans[2:],target[2:])/norm;ploss.backward();total+=float(ploss.detach());plans=plans.detach()
                    for ex,weights in zip(exs,l.category_weights(exs)):items.append((ex,plans[ex['beat']//4],role,goal,{k:v/norm for k,v in weights.items()}))
                    role_counts[role]+=len(exs)
            # Keep the statistical batch fixed; only execution microchunks shrink.
            for chunk in chunks(items,capacity):
                exs=[row[0] for row in chunk];deltas=[conditions(row[1],torch.zeros(1,8,device=model.device),row[2],row[3]) for row in chunk]
                values=er.batch_category_losses(model,exs,deltas,training=True)
                loss=sum(sum(v[k]*weight for k,weight in row[4].items()) for v,row in zip(values,chunk))
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                loss.backward();total+=float(loss.detach());count+=len(exs);batches.append(len(chunk))
        assert not any(p.grad is not None for name,p in model.named_parameters() if 'lora_' not in name)
        norm=torch.nn.utils.clip_grad_norm_(allparams,1.,error_if_nonfinite=True)
        warmup=max(1,math.ceil(steps*.05));factor=(step+1)/warmup if step<warmup else .5*(1+math.cos(math.pi*(step-warmup)/max(1,steps-warmup)))
        for group,lr in zip(optimizer.param_groups,[5e-5,1e-4]):group['lr']=lr*factor
        optimizer.step();state['step']=step+1
        row=dict(step=step+1,loss=total,examples=count,role_examples=role_counts,physical_microbatch=max(batches),seconds=time.monotonic()-tick,grad_norm=float(norm))
        state['updates'].append(row);print(json.dumps(row),flush=True)
        if (step+1)%32==0 or step+1==steps:
            scores=validate();passed=eligible(scores,baseline);state['validations'].append(dict(step=step+1,scores=scores,passed=passed))
            if passed and score(scores)<best:
                state['selected_step']=step+1;best=score(scores);atomic(out/'selected.pt',bundle())
        if (step+1)%16==0 or step+1==steps:persist()
    assert adapters.base_hash()==base_hash and sha(init)==init_sha
    state['complete']=state['step']==steps;state['frozen_backbone_unchanged']=True;state['orchestral_checkpoint_unchanged']=True
    state['status']='complete' if state['complete'] else 'time_limit';persist()
    return dict(output=str(out),step=state['step'],selected_step=state['selected_step'],complete=state['complete'])

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=128);p.add_argument('--hours',type=float,default=2);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    print(run(a.output,a.steps,a.hours,smoke=a.smoke))
