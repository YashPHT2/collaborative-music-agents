"""Bounded supervised phrase/revision run using the audited supported loader.

All three LoRA adapters, planner and partner conditioning learn. Backbone and
residual actors stay frozen. No PPO, DPO or audio quality reward is used.
"""
import argparse
import copy
from collections import Counter, OrderedDict
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

import supported_ensemble_loader as loader
import ensemble_repair as er
import offline_ensemble as oe
import supervised_ensemble as se
import note_trio as nt
from train_offline_ensemble import atomic
from train_ensemble_repair import write_json

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/supported_ensemble_run_v1'
SOURCE=ROOT/'outputs/offline_ensemble_v1/revised.pt'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dataset_checks(examples,groups,directory):
    from audit_supported_phrase_gate import validate_container
    for w in examples:
        validate_container(w['notes'],w['prefix_ticks'])
        if any(w['loss_beat_mask'][:w['loss_start_beat']]):raise ValueError('Prefix receives target loss')
        if w['unsupported_source_notes']:raise ValueError('Unsupported long note reached loader')
    counts={s:{r:len(groups[s][r]) for r in er.ROLES} for s in groups}
    write_json(directory/'loader_data_verification.json',dict(examples=len(examples),independent_compositions=counts,
        complete_commit_checks=True,prefix_loss_mask_verified=True,unsupported_notes_excluded=True,
        original_test_split_preserved=True,provenance_weights=dict(exact=1.,transfer=.35),minimum_note_filter=False))


def frozen_draft(model,adapters,agents,planner,conditions,w,start,seed):
    """Private partner proposals from the exact same real composition prefix."""
    ledger=nt.visible(w['notes'],start)
    with torch.no_grad(),oe.limits():
        plans=er.plan_sequence(planner,ledger,w['bars'])
        made=[]
        for role in er.ROLES:
            adapters.activate(role);own=copy.deepcopy(ledger)
            for beat in range(start,min(start+8,w['loss_end_beat'])):
                delta=conditions(plans[beat//4],torch.zeros(1,8,device=model.device),role)
                sampler=oe.Sampler(oe.ConditionedModel(model,delta),role,seed+beat*1009+er.ROLES.index(role)*100003,nt.Brief(w['bars']),beat,nt.visible(own,beat))
                notes,_=sampler.propose(agents[role],nt.state_features(nt.Brief(w['bars']),beat,nt.visible(own,beat)))
                own=nt.commit(own,{r:notes if r==role else [] for r in er.ROLES},beat,nt.Brief(w['bars']))
                made.extend(notes)
    return sorted(made,key=lambda n:(n[1],n[3],n[4]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT);parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--steps',type=int,default=128);parser.add_argument('--hours',type=float,default=1.)
    parser.add_argument('--microbatch',type=int,default=128);parser.add_argument('--phrases-per-role',type=int,default=4)
    parser.add_argument('--verify-only',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if not 1<=args.steps<=512 or not 0<args.hours<=3 or args.microbatch not in (8,16,32,64,128) or not 1<=args.phrases_per_role<=8:
        parser.error('Invalid bounded run settings')
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    if (root/'progress.pt').exists() and not args.resume:raise ValueError('Use --resume or a new run directory')
    torch.set_num_threads(4);torch.manual_seed(989100);random.seed(989100)
    examples,groups,data_manifest=loader.load_data(minimum=12)
    dataset_checks(examples,groups,root)
    source=torch.load(args.source,map_location='cpu',weights_only=False)
    model=nt.load_model();adapters=se.InstrumentAdapters(model);adapters.load(source['adapters'])
    if source.get('format_version')==2:planner,conditions=er.upgrade(source,model.device)
    else:
        planner=er.Planner().to(model.device);conditions=er.Conditions().to(model.device)
        planner.load_state_dict(source['planner']);conditions.load_state_dict(source['conditions'])
    base_hash=adapters.base_hash()
    if base_hash!=source['base_sha256']:raise ValueError('Source/frozen backbone mismatch')
    agents={r:oe.Musician().eval() for r in er.ROLES}
    params=[p for r in er.ROLES for p in adapters.params(r)]+list(planner.parameters())+list(conditions.parameters())
    optimizer=torch.optim.AdamW(params,lr=3e-5,weight_decay=.01)
    state=dict(stage=0,step=0,total=args.steps,elapsed=0.,started=False,logs=[],updates=[],complete=False,selections={},
               batch=dict(microbatch=args.microbatch,phrases_per_role=args.phrases_per_role,maximum_beat_examples_per_update=48*args.phrases_per_role))
    config=dict(source=str(args.source.resolve()),source_sha256=sha(args.source),data_sha256=data_manifest['eligible_examples_sha256'],
        steps_per_stage=args.steps,hours=args.hours,phrases_per_role=args.phrases_per_role,
        objective='Supervised masked category cross-entropy, equal role/group and onset/non-onset balance',
        trainable='Rank-8 organ/strings/cello LoRA + causal bar planner + partner-draft conditioning',
        frozen='SkyTNT backbone and residual actors',provenance_weights=dict(exact=1.,transfer=.35),
        stages=['participation','revision'],drafts='Frozen stage-start private proposals from the exact real prefix; fixed validation drafts; cache refreshed by stage')
    # Train-only profile metadata keeps format-3 generation compatible.
    quantiles={}
    representatives=[ws[0] for role in er.ROLES for ws in groups['train'][role].values()]
    values=torch.stack([er.description(w['notes'],b,w['bars']) for w in representatives for b in range(w['loss_start_beat']//4,w['bars'])])
    for index in range(12,21):
        observation=values[:,index]
        if index>=15:observation=observation[values[:,12+(index-15)%3]>0]
        quantiles[str(index)]=torch.quantile(observation,torch.tensor([.35,.65])).tolist()
    def bundle():
        return dict(format_version=3,plan_size=48,adapters=adapters.state(),planner=planner.state_dict(),conditions=conditions.state_dict(),
            constraints=dict(strings_cap=4,max_duration_ticks=3840),config=config,base_sha256=base_hash,
            profile_quantiles=quantiles,source=str(args.source.resolve()),composition_mode=state.get('composition_mode','planned'))
    def restore(saved):
        adapters.load(saved['adapters']);planner.load_state_dict(saved['planner']);conditions.load_state_dict(saved['conditions'])
    started=time.monotonic();prior=0.;draft_cache={};frozen=None
    if args.resume:
        saved=torch.load(root/'progress.pt',map_location='cpu',weights_only=False)
        if saved['config']!=config:raise ValueError('Resume recipe changed')
        restore(saved);optimizer.load_state_dict(saved['optimizer']);state=saved['state'];prior=state['elapsed']
        torch.set_rng_state(saved['rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
    def elapsed():return prior+time.monotonic()-started
    def save():
        state['elapsed']=elapsed()
        write_json(root/'training_log.json',state)
        atomic(root/'progress.pt',{**bundle(),'optimizer':optimizer.state_dict(),'state':state,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()})
    teacher_cache=OrderedDict()
    def teachers(w,role):
        key=w['id'],role
        if key not in teacher_cache:
            entries=[]
            for beat in range(w['loss_start_beat'],w['loss_end_beat']):
                ex=loader.teacher(model,w,role,beat)
                if ex is not None:entries.append(ex)
            teacher_cache[key]=entries
            if len(teacher_cache)>128:teacher_cache.popitem(last=False)
        else:teacher_cache.move_to_end(key)
        return teacher_cache[key]
    # Actual tokenizer/legal-mask checks, then one gradient-only probe per role.
    print('Verifying production teacher, sparse/sustain labels and gradients',flush=True)
    verification={};verification_start=copy.deepcopy(bundle())
    verification_rng=torch.get_rng_state();verification_cuda_rng=torch.cuda.get_rng_state_all()
    for role in er.ROLES:
        pool=[w for w in examples if w['split']=='train' and role in w['available_roles']]
        pool.sort(key=lambda w:(w['roles'][role]['state']!='present_sustaining',w['roles'][role]['state']!='present_sparse',w['id']))
        checked=0;gradient=False
        for w in pool[:8]:
            if loader.teacher(model,w,role,max(0,w['loss_start_beat']-1)) is not None:raise ValueError('Prefix produced teacher loss')
            exs=teachers(w,role);checked+=len(exs)
            if exs and not gradient:
                adapters.activate(role,training=True);planner.train();conditions.train()
                optimizer.zero_grad(set_to_none=True)
                plans=er.plan_sequence(planner,w['notes'],w['bars'])
                chunk=exs[:2];deltas=[conditions(plans[e['beat']//4],torch.zeros(1,8,device=model.device),role) for e in chunk]
                losses=er.batch_category_losses(model,chunk,deltas,training=True)
                loss=sum(sum(row.values())/max(1,len(row)) for row in losses)/len(losses)
                loss.backward()
                norm=sum(float(p.grad.float().square().sum()) for p in adapters.params(role) if p.grad is not None)**.5
                if not norm>0 or not np.isfinite(norm):raise ValueError('No finite adapter gradient')
                if any(p.grad is not None for n,p in model.named_parameters() if 'lora_' not in n):raise ValueError('Frozen backbone got gradients')
                gradient=True;optimizer.zero_grad(set_to_none=True)
        if not checked or not gradient:raise ValueError('No verified teacher decisions')
        verification[role]=dict(teacher_decisions_checked=checked,finite_nonzero_lora_gradient=True)
    if adapters.base_hash()!=base_hash:raise ValueError('Backbone changed during loader verification')
    # Probe uses no optimizer steps, so source weights stay the initialization.
    restore(verification_start)
    torch.set_rng_state(verification_rng);torch.cuda.set_rng_state_all(verification_cuda_rng)
    write_json(root/'loader_integration_verification.json',dict(passed=True,per_role=verification,
        prefix_loss_excluded=True,legal_sampling_support_verified=True,frozen_backbone_unchanged=True,
        data_sha256=config['data_sha256'],source_sha256=config['source_sha256'],optimizer_steps_during_verification=0))
    data_manifest.update(training_loader_integrated=True,supervised_training_allowed=True,
                         loader_verification=str((root/'loader_integration_verification.json').resolve()))
    write_json(loader.DATA/'summary.json',data_manifest)
    if args.verify_only:print('Verified loader integration complete; no training requested',flush=True);return
    data_manifest.update(training_started=True,training_run=str(root),training_recipe=config)
    write_json(loader.DATA/'summary.json',data_manifest)
    validations={r:loader.validation_set(groups['validation'],r) for r in er.ROLES}
    def draft(w,start):
        key=w['id'],start
        if key not in draft_cache:
            current=copy.deepcopy(bundle());restore(frozen);planner.eval();conditions.eval()
            draft_cache[key]=frozen_draft(model,adapters,agents,planner,conditions,w,start,989500+int(w['id'][:6],16))
            restore(current)
        return draft_cache[key]
    @torch.no_grad()
    def validate(stage):
        scores={};planner.eval();conditions.eval()
        for role in er.ROLES:
            adapters.activate(role);rows=[]
            for w in validations[role]:
                plans=er.plan_sequence(planner,w['notes'],w['bars']);entries=teachers(w,role)
                for offset in range(0,len(entries),16):
                    chunk=entries[offset:offset+16];deltas=[]
                    for ex in chunk:
                        segment=ex['beat']//8*8
                        partner=[n for n in draft(w,segment) if n[3]!=er.ROLES.index(role)] if stage=='revision' else []
                        deltas.append(conditions(plans[ex['beat']//4],oe.draft_features(partner,role,segment*480).to(model.device),role))
                    adapters.activate(role)
                    values=er.batch_category_losses(model,chunk,deltas)
                    for ex,result in zip(chunk,values):rows.append(dict(group=w['group'],kind=ex['kind'],provenance=ex['provenance'],losses={k:float(v) for k,v in result.items()}))
            scores[role]={'all':er.aggregate(rows)}
            for provenance in ('exact','transfer'):
                subset=[r for r in rows if r['provenance']==provenance]
                scores[role][provenance]=er.aggregate(subset) if subset else None
        return scores
    if not (root/'initial.pt').exists():atomic(root/'initial.pt',bundle())
    save();deadline=args.hours*3600;training_deadline=deadline*.85
    try:
        for stage_index in range(state['stage'],2):
            stage=['participation','revision'][stage_index];state['stage']=stage_index
            if (root/f'{stage}_start.pt').exists():frozen=torch.load(root/f'{stage}_start.pt',map_location='cpu',weights_only=False)
            else:frozen=copy.deepcopy(bundle());atomic(root/f'{stage}_start.pt',frozen)
            draft_cache={}
            if not state['started']:
                print('Baseline held-out validation:',stage,flush=True)
                state['status']='Baseline validation: '+stage
                state['elapsed']=elapsed();write_json(root/'training_log.json',state)
                state['initial']=validate(stage);state['best']=np.mean([state['initial'][r]['all']['event'] for r in er.ROLES])
                state['selected_step']=0;state['started']=True;state['stage_started_elapsed']=elapsed()
                atomic(root/f'{stage}.pt',bundle());save()
            for step in range(state['step'],args.steps):
                if elapsed()>=training_deadline or elapsed()-state['stage_started_elapsed']>=training_deadline/2:break
                rng=random.Random(989100+stage_index*100000+step);update_start=time.monotonic()
                optimizer.zero_grad(set_to_none=True);total=0.;used=0;by_role={};largest_batch=0
                state['status']='Training: '+stage
                # Revision rotates a hash-ranked group subset every 16 updates;
                # this bounds on-demand generation without using first-12 windows.
                shared_pool=sorted(groups['train'][er.ROLES[0]])
                if stage=='revision':
                    random.Random(989800+step//16).shuffle(shared_pool);shared_pool=shared_pool[:12]
                for role in er.ROLES:
                    items=[];role_loss=0.
                    for phrase in range(args.phrases_per_role):
                        pool=shared_pool if stage=='revision' else sorted(groups['train'][role])
                        group=rng.choice(pool);w=rng.choice(groups['train'][role][group])
                        entries=teachers(w,role)
                        if not entries:continue
                        # Generate/cache matched private drafts before creating a
                        # live graph; restoration never mutates graph parameters.
                        drafts={segment:draft(w,segment) for segment in range(w['loss_start_beat'],w['loss_end_beat'],8)} if stage=='revision' else {}
                        adapters.activate(role,training=True);planner.train();conditions.train()
                        plans=er.plan_sequence(planner,w['notes'],w['bars'])
                        target=torch.stack([er.description(w['notes'],b,w['bars']) for b in range(w['bars'])]).to(model.device)
                        start_bar=w['loss_start_beat']//4
                        ploss=.1*F.mse_loss(plans[start_bar:],target[start_bar:])*w['role_weights'][role]/(3*args.phrases_per_role)
                        ploss.backward();total+=float(ploss.detach());role_loss+=float(ploss.detach());plans=plans.detach()
                        weights=loader.balance_weights(entries,w['role_weights'][role],3*args.phrases_per_role)
                        for ex,weight in zip(entries,weights):
                            segment=ex['beat']//8*8
                            partner=[n for n in drafts.get(segment,[]) if n[3]!=er.ROLES.index(role)]
                            items.append((ex,plans[ex['beat']//4],oe.draft_features(partner,role,segment*480).to(model.device),weight))
                    # Context-length sorting only changes padding, not examples.
                    items.sort(key=lambda x:len(x[0]['x']));offset=0
                    while offset<len(items):
                        batch=state['batch']['microbatch'];chunk=items[offset:offset+batch]
                        largest_batch=max(largest_batch,len(chunk))
                        try:
                            deltas=[conditions(row[1],row[2],role) for row in chunk]
                            values=er.batch_category_losses(model,[row[0] for row in chunk],deltas,training=True)
                            loss=sum(sum(v[k]*weight for k,weight in row[3].items()) for v,row in zip(values,chunk))
                            if not torch.isfinite(loss):raise ValueError('Nonfinite supervised loss')
                            loss.backward()
                        except torch.cuda.OutOfMemoryError:
                            # A failed backward can have partial gradients; do not
                            # retry this update with contaminated accumulation.
                            raise RuntimeError('CUDA OOM: checkpoint retained; lower --microbatch on a fresh verified run')
                        total+=float(loss.detach());role_loss+=float(loss.detach());used+=len(chunk);offset+=len(chunk)
                    by_role[role]=role_loss
                if not used:raise ValueError('No supported target decisions')
                torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step()
                state['step']=step+1;row=dict(stage=stage,step=step+1,loss=total,role_loss=by_role,examples=used,largest_physical_microbatch=largest_batch,seconds=time.monotonic()-update_start)
                state['updates'].append(row);state['last_update']=row
                print(f'{stage} {step+1}/{args.steps}: loss={total:.5f} examples={used} role_losses={by_role}',flush=True)
                # Always expose every actual optimizer update to the live panel.
                state['elapsed']=elapsed();write_json(root/'training_log.json',state)
                if (step+1)%16==0 or step+1==args.steps:
                    scores=validate(stage);eligible=loader.selection_eligible(scores,state['initial'])
                    mean=float(np.mean([scores[r]['all']['event'] for r in er.ROLES]))
                    state['logs'].append(dict(stage=stage,step=step+1,validation=scores,eligible=eligible))
                    if eligible and mean<state['best']:
                        state['best']=mean;state['selected_step']=step+1
                        state['composition_mode']='revised' if stage=='revision' else 'planned'
                        atomic(root/f'{stage}.pt',bundle())
                    save()
                elif (step+1)%4==0:save()
            atomic(root/f'{stage}_candidate.pt',bundle())
            chosen=torch.load(root/f'{stage}.pt',map_location='cpu',weights_only=False);restore(chosen)
            state['composition_mode']=chosen.get('composition_mode','planned')
            state['selections'][stage]=dict(step=state['selected_step'],passed=state['selected_step']>0)
            state.update(stage=stage_index+1,step=0,started=False);save()
            if elapsed()>=training_deadline:break
        atomic(root/'selected.pt',bundle())
        if adapters.base_hash()!=base_hash:raise ValueError('Frozen backbone changed')
        state['complete']=True;state['status']='Bounded supervised run ended; listening quality untested';save()
        print('Supervised run finished; selections:',state['selections'],flush=True)
    except BaseException as error:
        state['status']=str(error);save();raise


if __name__=='__main__':main()
