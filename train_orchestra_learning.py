"""Bounded, masked supervised ensemble training for all orchestral role slots."""
import os
os.environ.setdefault('USE_TF','0')
os.environ.setdefault('USE_FLAX','0')
import argparse
from collections import OrderedDict,defaultdict
import copy
import gzip
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import orchestra_learning as l
import note_trio as nt
import offline_ensemble as oe
import ensemble_repair as er
from train_ensemble_repair import write_json
from train_offline_ensemble import atomic

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/orchestra_learning_v1'
DATA=OUT/'data'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def macro(scores):return float(np.mean([np.mean([scores[r]['event'] for r,s in l.ROLES.items() if s['family']==family]) for family in l.FAMILIES]))
def eligible(scores,initial):
    return macro(scores)<macro(initial)*.995 and all(scores[r]['event']<=initial[r]['event']*1.02 and
        scores[r]['attributes'] is not None and scores[r]['attributes']<=initial[r]['attributes']*1.02 for r in l.ROLES)


def load_data():
    summary=json.loads((DATA/'summary.json').read_text())
    if not summary.get('meter_verified') or not summary['validation_gate_passed']:raise ValueError('Verified data gate failed')
    if sha(DATA/'examples.json.gz')!=summary['examples_sha256']:raise ValueError('Data manifest hash mismatch')
    with gzip.open(DATA/'examples.json.gz','rt') as f:windows=json.load(f)
    groups={s:{r:defaultdict(list) for r in l.ROLES} for s in ['train','validation','test']}
    for w in windows:
        for role in w['available_roles']:groups[w['split']][role][w['group']].append(w)
    splits={s:{w['group'] for w in windows if w['split']==s} for s in groups}
    assert not(splits['train']&splits['validation'] or splits['train']&splits['test'] or splits['validation']&splits['test'])
    assert all(len(groups['validation'][r])>=12 and len(groups['train'][r])>=12 for r in l.ROLES)
    return windows,groups,summary


def fixed_windows(groups,role,limit=12):
    keys=sorted(groups[role],key=lambda g:hashlib.sha256((role+g).encode()).hexdigest())[:limit];result=[]
    for group in keys:
        pool=groups[role][group]
        active=[w for w in pool if any(n[3]==l.ROLES[role]['id'] and n[1]>=3840 for n in w['notes'])]
        result.append(min(active or pool,key=lambda w:w['id']))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--steps',type=int,default=64)
    parser.add_argument('--microbatch',type=int,default=128);parser.add_argument('--phrases-per-role',type=int,default=2)
    parser.add_argument('--hours',type=float,default=1.5);args=parser.parse_args()
    if not 1<=args.steps<=256 or args.microbatch not in [16,32,64,128] or not 1<=args.phrases_per_role<=2 or not 0<args.hours<=3:parser.error('Invalid bounded recipe')
    if (OUT/'progress.pt').exists():raise ValueError('Existing run retained; use a new version for another recipe')
    torch.set_num_threads(4);torch.manual_seed(9911001);random.seed(9911001)
    windows,groups,data=load_data();model=nt.load_model();adapters=l.Adapters(model)
    planner=l.Planner().to(model.device);conditions=l.Conditions().to(model.device)
    agents={r:oe.Musician().eval() for r in l.ROLES};base_hash=adapters.base_hash()
    params=[p for f in l.FAMILIES for p in adapters.params(f)]+list(planner.parameters())+list(conditions.parameters())
    initial_adapters=copy.deepcopy(adapters.state());initial_conditions=copy.deepcopy(conditions.state_dict())
    config=dict(steps_per_stage=args.steps,microbatch_capacity=args.microbatch,phrases_per_role=args.phrases_per_role,hours=args.hours,
                data_sha256=data['examples_sha256'],family_adapters=7,instrument_role_slots=19,source='Original pretrained SkyTNT',
                objective='Balanced masked note categories + causal four-bar phrase planning; no PPO or taste reward.',
                role_sampling='Equal roles then compositions, masked missing roles, one-to-one real track ownership per phrase.',
                partner_stage='75% real announced partner parts; 25% frozen generated drafts on fixed role-balanced anchors from the same prefix.',
                validation='Twelve fixed composition groups per role; real announced partner drafts in partner-stage teacher validation, not free-running cooperation.',
                selection='At least 0.5% lower family-macro event loss, no role event or attribute regression above 2%.',
                focus='Measured low-density/velocity controls and C-major pitch mask; no human mood labels or concentration benefit claim.')
    state=dict(status='Verifying teacher integration',stage='verification',step=0,total=args.steps,updates=[],validations=[],selections={},elapsed=0.,complete=False,
               batch=dict(capacity=args.microbatch,phrases_per_role=args.phrases_per_role),config=config)
    start_time=time.monotonic()
    def elapsed():return time.monotonic()-start_time
    def status(message):state['status']=message;state['elapsed']=elapsed();write_json(OUT/'training_log.json',state)
    def bundle(mode='planned'):return dict(format_version=4,roles=l.ROLES,families=l.FAMILIES,adapters=adapters.state(),planner=planner.state_dict(),
        conditions=conditions.state_dict(),config=config,base_sha256=base_hash,composition_mode=mode,calm_profile=calm_profile)
    def restore(saved):adapters.load(saved['adapters']);planner.load_state_dict(saved['planner']);conditions.load_state_dict(saved['conditions'])
    cache=OrderedDict()
    def teachers(w,role):
        key=w['id'],role
        if key not in cache:
            examples=[ex for beat in range(8,24) if (ex:=l.teacher(model,w,role,beat)) is not None]
            cache[key]=examples
            if len(cache)>96:cache.popitem(last=False)
        else:cache.move_to_end(key)
        return cache[key]
    validations={r:fixed_windows(groups['validation'],r) for r in l.ROLES}
    # Profile quantiles use only training compositions, one phrase/group and role.
    density=[];velocity=[]
    for role,s in l.ROLES.items():
        rows=[]
        for group,pool in groups['train'][role].items():
            w=min(pool,key=lambda w:w['id'])
            for bar in range(2,6):
                own=[n for n in w['notes'] if n[3]==s['id'] and bar*1920<=n[1]<(bar+1)*1920]
                if own:rows.append((len(own)/16,np.mean([n[5] for n in own])/127))
        density.append(float(np.quantile([v[0] for v in rows],.35)));velocity.append(float(np.quantile([v[1] for v in rows],.35)))
    calm_profile=dict(density=density,velocity=velocity,style=[float(np.mean(density)),float(np.mean(velocity)),.4,1.])
    verification={};status('Verifying real-model legal targets and gradients for all 19 roles')
    for role,s in l.ROLES.items():
        family=s['family'];adapters.activate(family,True);conditions.train();planner.train()
        chosen=next((w for g in sorted(groups['train'][role]) for w in groups['train'][role][g]
                     if any(n[3]==s['id'] and n[1]>=3840 for n in w['notes'])),None)
        if chosen is None:raise ValueError('No real target notes for '+role)
        assert l.teacher(model,chosen,role,7) is None
        exs=teachers(chosen,role);active=[ex for ex in exs if ex['kind']=='onset'];chunk=(active or exs)[:2]
        if not chunk:raise ValueError('No supported teacher for '+role)
        goal=torch.tensor(chosen['style'],device=model.device);plans=l.plan_sequence(planner,chosen['notes'],6,chosen['available_roles'],goal)
        values=er.batch_category_losses(model,chunk,[conditions(plans[ex['beat']//4],torch.zeros(1,8,device=model.device),role,goal) for ex in chunk],training=True)
        loss=sum(sum(v.values())/len(v) for v in values)/len(values);loss.backward()
        norm=sum(float(p.grad.float().square().sum()) for p in adapters.params(family) if p.grad is not None)**.5
        if not np.isfinite(norm) or norm<=0:raise ValueError('No finite family adapter gradient for '+role)
        if any(p.grad is not None for n,p in model.named_parameters() if 'lora_' not in n):raise ValueError('Frozen backbone received gradients')
        for p in params:p.grad=None
        verification[role]=dict(teacher_beats=len(exs),onset_beats=len(active),family_adapter_gradient_norm=norm,
                                validation_groups=len(groups['validation'][role]),prefix_loss_excluded=True)
        print('Verified teacher and family gradient:',role,flush=True)
    if adapters.base_hash()!=base_hash:raise ValueError('Backbone changed in preflight')
    write_json(OUT/'loader_verification.json',dict(passed=True,roles=verification,no_optimizer_steps_in_verification=True,
        missing_roles_masked=True,prefix_loss_excluded=True,backbone_unchanged=True,data_sha256=data['examples_sha256']))
    atomic(OUT/'initial.pt',bundle());data.update(training_started=True,training_run=str(OUT));write_json(DATA/'summary.json',data)
    @torch.no_grad()
    def validate(stage):
        planner.eval();conditions.eval();scores={}
        for role,s in l.ROLES.items():
            adapters.activate(s['family']);rows=[]
            for w in validations[role]:
                goal=torch.tensor(w['style'],device=model.device);plans=l.plan_sequence(planner,w['notes'],6,w['available_roles'],goal);entries=teachers(w,role)
                for offset in range(0,len(entries),16):
                    chunk=entries[offset:offset+16];deltas=[]
                    for ex in chunk:
                        segment=ex['beat']//8*8
                        partner=[n for n in w['notes'] if n[3]!=s['id'] and segment*480<=n[1]<(segment+8)*480] if stage=='partner' else []
                        deltas.append(conditions(plans[ex['beat']//4],l.draft_features(partner,role,segment).to(model.device),role,goal))
                    values=er.batch_category_losses(model,chunk,deltas)
                    rows.extend(dict(group=w['group'],kind=ex['kind'],losses={k:float(v) for k,v in value.items()}) for ex,value in zip(chunk,values))
            scores[role]=er.aggregate(rows)
        return scores
    try:
        for stage_index,stage in enumerate(['phrases','partner']):
            state.update(stage=stage,step=0);optimizer=torch.optim.AdamW(params,lr=3e-5,weight_decay=.01)
            anchors={r:fixed_windows(groups['train'],r,1)[0] for r in l.ROLES};drafts={}
            if stage=='partner':
                planner.eval();conditions.eval();directory=OUT/'drafts';directory.mkdir(exist_ok=True)
                unique={w['id']:w for w in anchors.values()}
                for i,w in enumerate(unique.values()):
                    status(f'Preparing frozen partner drafts {i+1}/{len(unique)}')
                    goal=torch.tensor(w['style'],device=model.device)
                    for segment in (8,16):
                        drafts[w['id'],segment]=l.private_draft(model,adapters,agents,planner,conditions,l.visible(w['notes'],segment),segment,6,w['available_roles'],goal,9921001+i*31)
                    write_json(directory/(w['id']+'.json'),dict(window_id=w['id'],source_group=w['group'],seed=9921001+i*31,
                        drafts={str(k):drafts[w['id'],k] for k in (8,16)},own_future_targets_used=False))
            status('Baseline held-out validation: '+stage);baseline=validate(stage)
            state['initial']=baseline;state['selected_step']=0;best=macro(baseline);mode='planned' if stage=='phrases' else 'revised'
            atomic(OUT/(stage+'.pt'),bundle('planned' if stage=='phrases' else 'planned'))
            for step in range(args.steps):
                if elapsed()>args.hours*3600*.85:state['budget_stopped']=True;break
                tick=time.monotonic();retry=True
                while retry:
                    retry=False;optimizer.zero_grad(set_to_none=True);rng=random.Random(9922001+stage_index*100000+step)
                    total=0.;count=0;family_losses={};largest=0;role_examples={};planner.train();conditions.train()
                    try:
                        for family in l.FAMILIES:
                            adapters.activate(family,True);items=[];family_loss=0.;roles=[r for r,s in l.ROLES.items() if s['family']==family]
                            for role in roles:
                                role_count=0
                                for phrase in range(args.phrases_per_role):
                                    generated=stage=='partner' and (step+l.ROLES[role]['id']+phrase)%4==0
                                    if generated:w=anchors[role]
                                    else:
                                        group=rng.choice(sorted(groups['train'][role]));w=rng.choice(groups['train'][role][group])
                                    entries=teachers(w,role)
                                    if not entries:continue
                                    goal=torch.tensor(w['style'],device=model.device);plans=l.plan_sequence(planner,w['notes'],6,w['available_roles'],goal)
                                    target=torch.stack([l.description(w['notes'],bar,6,w['available_roles']) for bar in range(6)]).to(model.device)
                                    normalizer=len(l.FAMILIES)*len(roles)*args.phrases_per_role;source_weight=w['role_weights'][role]
                                    ploss=.1*F.mse_loss(plans[2:],target[2:])*source_weight/normalizer
                                    ploss.backward();value=float(ploss.detach());total+=value;family_loss+=value;plans=plans.detach()
                                    for ex,weights in zip(entries,l.category_weights(entries)):
                                        segment=ex['beat']//8*8
                                        announced=drafts[w['id'],segment] if generated else [n for n in w['notes'] if segment*480<=n[1]<(segment+8)*480]
                                        partner=[n for n in announced if n[3]!=l.ROLES[role]['id']] if stage=='partner' else []
                                        items.append((ex,plans[ex['beat']//4],l.draft_features(partner,role,segment),role,goal,
                                                      {k:v*source_weight/normalizer for k,v in weights.items()}))
                                    role_count+=len(entries)
                                role_examples[role]=role_count
                            items.sort(key=lambda row:len(row[0]['x']))
                            for offset in range(0,len(items),args.microbatch):
                                chunk=items[offset:offset+args.microbatch];largest=max(largest,len(chunk))
                                values=er.batch_category_losses(model,[row[0] for row in chunk],
                                    [conditions(row[1],row[2].to(model.device),row[3],row[4]) for row in chunk],training=True)
                                loss=sum(sum(result[k]*weight for k,weight in row[5].items()) for result,row in zip(values,chunk))
                                if not torch.isfinite(loss):raise ValueError('Nonfinite orchestral training loss')
                                loss.backward();value=float(loss.detach());total+=value;family_loss+=value;count+=len(chunk)
                            family_losses[family]=family_loss
                    except torch.cuda.OutOfMemoryError:
                        if args.microbatch<=16:raise
                        optimizer.zero_grad(set_to_none=True);torch.cuda.empty_cache();args.microbatch//=2
                        state['batch']['capacity']=args.microbatch;status('Retrying full optimizer update with smaller physical batch');retry=True
                torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step()
                row=dict(stage=stage,step=step+1,loss=total,examples=count,role_examples=role_examples,
                         family_losses=family_losses,physical_microbatch=largest,seconds=time.monotonic()-tick)
                state['step']=step+1;state['updates'].append(row);status('Training: '+stage)
                print(f'{stage} {step+1}/{args.steps}: loss={total:.5f} examples={count} microbatch={largest}',flush=True)
                if (step+1)%16==0 or step+1==args.steps:
                    status('Held-out validation: '+stage);scores=validate(stage);passed=eligible(scores,baseline)
                    state['validations'].append(dict(stage=stage,step=step+1,scores=scores,passed=passed))
                    if passed and macro(scores)<best:
                        best=macro(scores);state['selected_step']=step+1;atomic(OUT/(stage+'.pt'),bundle(mode))
                    atomic(OUT/'progress.pt',{**bundle(mode), 'state':copy.deepcopy(state),'optimizer':optimizer.state_dict()})
                elif (step+1)%8==0:atomic(OUT/'progress.pt',{**bundle(mode),'state':copy.deepcopy(state),'optimizer':optimizer.state_dict()})
            atomic(OUT/(stage+'_candidate.pt'),bundle(mode));state['selections'][stage]=dict(step=state['selected_step'],passed=state['selected_step']>0)
            chosen=torch.load(OUT/(stage+'.pt'),map_location='cpu',weights_only=False);restore(chosen)
            state['composition_mode']=chosen['composition_mode']
            if state.get('budget_stopped'):break
        # Selected metrics and candidate training progress are separate records.
        selected=bundle(state.get('composition_mode','planned'));atomic(OUT/'selected.pt',selected)
        if adapters.base_hash()!=base_hash:raise ValueError('Frozen backbone changed')
        differences={f:sum(float((v-initial_adapters[k]).float().square().sum()) for k,v in adapters.state().items() if f'.{f}.' in k)**.5 for f in l.FAMILIES}
        role_delta=(conditions.roles.weight.detach().cpu()-initial_conditions['roles.weight'].cpu()).norm(dim=1)
        write_json(OUT/'parameter_changes.json',dict(selected_family_lora_l2=differences,
            selected_role_embedding_l2={r:float(role_delta[s['id']]) for r,s in l.ROLES.items()},backbone_unchanged=True))
        state['complete']=True;status('Supervised training complete; musical enjoyment untested')
        print('TRAINING COMPLETE',state['selections'],flush=True)
    except BaseException as error:
        state['failure']=str(error);status('Training interrupted: '+str(error));raise


if __name__=='__main__':main()
