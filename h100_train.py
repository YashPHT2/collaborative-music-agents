"""Portable BF16 notebook trainer with real batch calibration and exact resume.

Keeps the audited four-bar targets. Extra VRAM does not create missing long
scores or mood labels. All 19 roles train through seven family adapters.
"""
import os
os.environ.setdefault('USE_TF','0')
os.environ.setdefault('USE_FLAX','0')
import argparse
from collections import OrderedDict
import copy
import gc
import hashlib
import json
from pathlib import Path
import random
import signal
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import orchestra_learning as l
import ensemble_repair as er
import note_trio as nt
import offline_ensemble as oe
from train_orchestra_learning import load_data,fixed_windows,eligible,macro
from train_offline_ensemble import atomic
from train_ensemble_repair import write_json

ROOT=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rng_state():return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),python=random.getstate(),numpy=np.random.get_state())
def restore_rng(saved):
    torch.set_rng_state(saved['torch']);torch.cuda.set_rng_state_all(saved['cuda']);random.setstate(saved['python']);np.random.set_state(saved['numpy'])


def microchunks(items,capacity,token_budget):
    """Cap real examples and padded event tokens; never discard examples."""
    ordered=sorted(items,key=lambda row:len(row[0]['x']));chunk=[];maximum=0
    for row in ordered:
        length=len(row[0]['x'])
        if length>token_budget:raise ValueError('One sequence exceeds padded-token budget')
        if chunk and (len(chunk)>=capacity or max(maximum,length)*(len(chunk)+1)>token_budget):
            yield chunk;chunk=[];maximum=0
        chunk.append(row);maximum=max(maximum,length)
    if chunk:yield chunk


def safe_load(path):return torch.load(path,map_location='cpu',weights_only=False)


def no_role_regression(scores,baseline,tolerance=.02):
    for role in l.ROLES:
        for category in ['event','attributes']:
            current=scores[role][category];before=baseline[role][category]
            if current is None or before is None or not np.isfinite(current) or not np.isfinite(before):return False
            if current>before*(1+tolerance):return False
    return True


class Runner:
    def __init__(self,args):
        self.args=args;self.out=args.output.resolve();self.out.mkdir(parents=True,exist_ok=True)
        self.started=time.monotonic();self.prior=0.;self.stop=False
        torch.set_num_threads(args.cpu_threads);torch.manual_seed(args.seed);random.seed(args.seed);np.random.seed(args.seed)
        if not torch.cuda.is_available():raise RuntimeError('A CUDA notebook GPU is required')
        gpu=torch.cuda.get_device_properties(0)
        if not args.allow_non_h100 and ('H100' not in gpu.name or gpu.total_memory<70*1024**3):
            raise RuntimeError('Select an H100 80GB runtime, or explicitly use --allow-non-h100 for local verification')
        if not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 GPU support is required')
        torch.backends.cuda.matmul.allow_tf32=True
        self.windows,self.groups,self.data=load_data()
        self.model=nt.load_model();self.adapters=l.Adapters(self.model)
        if not args.gradient_checkpointing:self.model.net.gradient_checkpointing_disable()
        self.planner=l.Planner().to(self.model.device);self.conditions=l.Conditions().to(self.model.device)
        self.agents={r:oe.Musician().eval() for r in l.ROLES};self.base_hash=self.adapters.base_hash()
        self.params=[p for f in l.FAMILIES for p in self.adapters.params(f)]+list(self.planner.parameters())+list(self.conditions.parameters())
        self.teacher_cache=OrderedDict();self.active_snapshot=None
        self.config=dict(data_sha256=self.data['examples_sha256'],steps=args.steps,phrases_per_role=args.phrases_per_role,
            learning_rate=args.learning_rate,seed=args.seed,validation_groups=args.validation_groups,
            validation_interval=args.validation_interval,generated_fraction=.25,draft_refresh=args.draft_refresh,
            gradient_checkpointing=args.gradient_checkpointing,precision='bf16-backbone-fp32-adapters',
            init_sha256=sha(args.init),target_bars=4,prefix_bars=2,format_version=1)
        if args.start_stage=='partner':self.config['start_stage']='partner'
        if args.partner_warmup:self.config['partner_warmup']=args.partner_warmup
        if args.preserve_phrase:self.config['preserve_phrase']=True
        self.state=dict(stage_index=0,stage='phrases',step=0,total=args.steps,elapsed=0.,updates=[],validations=[],selections={},
            complete=False,status='Loading audited data',physical_capacity=args.microbatch or 128,token_budget=args.token_budget,
            draft_block=None,stage_ready=False,selected_mode='planned',selected_step=0,initial=None,best=None)
        if args.start_stage=='partner':self.state.update(stage_index=1,stage='partner')
        if args.resume:
            checkpoint=safe_load(self.out/'progress.pt')
            if checkpoint['h100_config']!=self.config:raise ValueError('Resume recipe/data/init mismatch; restore the original config')
            self.restore(checkpoint);self.state=checkpoint['state'];self.prior=self.state['elapsed']
            # Capacity is a runtime limit, not part of the statistical recipe.
            self.state['token_budget']=min(self.state['token_budget'],args.token_budget)
            if args.microbatch:self.state['physical_capacity']=args.microbatch
            self.profile=checkpoint['calm_profile'];self.optimizer=self.make_optimizer()
            if checkpoint.get('optimizer') is not None:self.optimizer.load_state_dict(checkpoint['optimizer'])
            restore_rng(checkpoint['rng'])
            if self.state['complete']:print('Run already complete',flush=True)
        else:
            if (self.out/'progress.pt').exists():raise ValueError('Existing run; use --resume or a new output folder')
            initial=safe_load(args.init)
            if initial.get('format_version')!=4 or initial['base_sha256']!=self.base_hash or initial['roles']!=l.ROLES:raise ValueError('Initialization is incompatible')
            self.restore(initial);self.profile=initial['calm_profile'];self.optimizer=self.make_optimizer()
            atomic(self.out/'initial.pt',self.bundle('planned'));self.save()
        self.validations={r:fixed_windows(self.groups['validation'],r,args.validation_groups) for r in l.ROLES}
        self.lookup={w['id']:w for w in self.windows}
        write_json(self.out/'environment.json',dict(python=sys.version,torch=torch.__version__,gpu=gpu.name,
            vram_gib=gpu.total_memory/1024**3,driver_cuda=torch.version.cuda,seed=args.seed,
            backbone_attention=self.model.net.config._attn_implementation,
            token_attention=self.model.net_token.config._attn_implementation,
            h100_executed='H100' in gpu.name))

    def make_optimizer(self):
        return torch.optim.AdamW(self.params,lr=self.args.learning_rate,weight_decay=.01,fused=self.model.device.type=='cuda')
    def elapsed(self):return self.prior+time.monotonic()-self.started
    def status(self,message):
        self.state.update(status=message,elapsed=self.elapsed());write_json(self.out/'training_log.json',self.state)
    def bundle(self,mode=None):
        return dict(format_version=4,roles=l.ROLES,families=l.FAMILIES,adapters=self.adapters.state(),
            planner=self.planner.state_dict(),conditions=self.conditions.state_dict(),base_sha256=self.base_hash,
            calm_profile=self.profile,composition_mode=mode or self.state['selected_mode'],h100_config=self.config)
    def restore(self,saved):
        self.adapters.load(saved['adapters']);self.planner.load_state_dict(saved['planner']);self.conditions.load_state_dict(saved['conditions'])
    def save(self):
        self.state['elapsed']=self.elapsed()
        atomic(self.out/'progress.pt',{**self.bundle(),'h100_config':self.config,'state':copy.deepcopy(self.state),
            'optimizer':self.optimizer.state_dict(),'rng':rng_state()})
        write_json(self.out/'training_log.json',self.state)
    def teachers(self,w,role):
        key=w['id'],role
        if key not in self.teacher_cache:
            self.teacher_cache[key]=[ex for beat in range(8,24) if (ex:=l.teacher(self.model,w,role,beat)) is not None]
            if len(self.teacher_cache)>self.args.cache_entries:self.teacher_cache.popitem(last=False)
        else:self.teacher_cache.move_to_end(key)
        return self.teacher_cache[key]
    def clear_grads(self):
        for p in self.params:p.grad=None

    def verify(self):
        self.status('Verifying legal note targets, gradients, and boundary sampling for 19 roles')
        rng=rng_state();report={}
        try:
            for role,s in l.ROLES.items():
                family=s['family'];self.adapters.activate(family,True);self.planner.train();self.conditions.train();self.clear_grads()
                chosen=next(w for group in sorted(self.groups['train'][role]) for w in self.groups['train'][role][group]
                    if any(n[3]==s['id'] and n[1]>=3840 for n in w['notes']))
                assert l.teacher(self.model,chosen,role,7) is None
                exs=self.teachers(chosen,role);chunk=[ex for ex in exs if ex['kind']=='onset'][:2]
                if not chunk:raise ValueError('No supported onset for '+role)
                goal=torch.tensor(chosen['style'],device=self.model.device)
                plans=l.plan_sequence(self.planner,chosen['notes'],6,chosen['available_roles'],goal)
                values=er.batch_category_losses(self.model,chunk,[self.conditions(plans[ex['beat']//4],torch.zeros(1,8,device=self.model.device),role,goal) for ex in chunk],training=True)
                loss=sum(sum(v.values())/len(v) for v in values)/len(values);loss.backward()
                norm=sum(float(p.grad.float().square().sum()) for p in self.adapters.params(family) if p.grad is not None)**.5
                assert np.isfinite(norm) and norm>0
                assert not any(p.grad is not None for n,p in self.model.named_parameters() if 'lora_' not in n)
                report[role]=dict(teacher_beats=len(exs),gradient_norm=norm,validation_groups=len(self.groups['validation'][role]))
                self.clear_grads();print('Verified',role,flush=True)
            self.adapters.activate('organ');self.conditions.eval();self.planner.eval()
            sampler=l.Sampler(self.model,'pipe_organ',self.args.seed,nt.Brief(6,70,0,False),23,[])
            assert 15 not in sampler.onsets()
            made,_=sampler.propose(self.agents['pipe_organ'],torch.zeros(32))
            assert all(n[1]+n[2]<=11520 for n in made)
            assert self.adapters.base_hash()==self.base_hash
            write_json(self.out/'verification.json',dict(passed=True,per_role=report,no_optimizer_steps=True,
                end_boundary_rollout_passed=True,frozen_backbone_unchanged=True,data_sha256=self.data['examples_sha256']))
        finally:self.clear_grads();restore_rng(rng)

    def calibrate(self):
        """Real forward/backward on long representative contexts, no optimizer steps."""
        self.status('Benchmarking physical batches on representative long contexts')
        rng=rng_state();rows=[]
        # Bound CPU tokenization before selecting the longest available contexts.
        role='piano_anchor';pool=fixed_windows(self.groups['train'],role,12)
        examples=[ex for w in pool for ex in self.teachers(w,role)]
        examples=sorted(examples,key=lambda ex:len(ex['x']),reverse=True)[:16]
        if not examples:raise ValueError('No calibration examples')
        self.adapters.activate('pianos',True);self.conditions.train();available=torch.cuda.get_device_properties(0).total_memory
        candidates=[16,32,64,128,256,512] if not self.args.microbatch else [self.args.microbatch]
        try:
            for size in candidates:
                if size*max(len(ex['x']) for ex in examples)>self.args.token_budget:continue
                try:
                    durations=[];peak=0
                    for repeat in range(3):
                        self.clear_grads();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.perf_counter()
                        exs=[examples[i%len(examples)] for i in range(size)]
                        plan=torch.zeros(l.SIZE,device=self.model.device);goal=torch.zeros(4,device=self.model.device)
                        deltas=[self.conditions(plan,torch.zeros(1,8,device=self.model.device),role,goal) for _ in exs]
                        values=er.batch_category_losses(self.model,exs,deltas,training=True)
                        loss=sum(sum(v.values())/len(v) for v in values)/len(values);loss.backward();torch.cuda.synchronize()
                        durations.append(time.perf_counter()-tick);peak=max(peak,torch.cuda.max_memory_allocated())
                        del loss,values,deltas,exs
                    seconds=float(np.median(durations[1:]));row=dict(batch=size,seconds=seconds,examples_per_second=size/seconds,
                        peak_allocated_gib=peak/1024**3,memory_fraction=peak/available,context_events=max(len(e['x']) for e in examples))
                    rows.append(row);print('Batch benchmark',row,flush=True)
                    if peak/available>.85:break
                except torch.cuda.OutOfMemoryError:
                    rows.append(dict(batch=size,oom=True));break
                finally:self.clear_grads();gc.collect();torch.cuda.empty_cache()
            safe=[r for r in rows if not r.get('oom') and r['memory_fraction']<=.85]
            if not safe:raise RuntimeError('No safe benchmark batch. Lower --microbatch / --token-budget or enable --gradient-checkpointing')
            fastest=max(r['examples_per_second'] for r in safe)
            selected=min((r for r in safe if r['examples_per_second']>=fastest*.95),key=lambda r:r['batch'])
            self.state['physical_capacity']=selected['batch']
            write_json(self.out/'batch_benchmark.json',dict(measurements=rows,selected_batch=selected['batch'],
                selection='Smallest within 5% of best throughput, <=85% allocated VRAM; no optimizer updates',h100_measured='H100' in torch.cuda.get_device_name()))
            restore_rng(rng)
            self.save()
        finally:self.clear_grads();restore_rng(rng)

    @torch.no_grad()
    def validate(self,stage):
        self.planner.eval();self.conditions.eval();scores={}
        for role,s in l.ROLES.items():
            self.adapters.activate(s['family']);rows=[]
            for w in self.validations[role]:
                goal=torch.tensor(w['style'],device=self.model.device);plans=l.plan_sequence(self.planner,w['notes'],6,w['available_roles'],goal)
                entries=self.teachers(w,role)
                capacity=min(32,self.state['physical_capacity'])
                for offset in range(0,len(entries),capacity):
                    chunk=entries[offset:offset+capacity];deltas=[]
                    for ex in chunk:
                        segment=ex['beat']//8*8
                        partner=[n for n in w['notes'] if n[3]!=s['id'] and segment*480<=n[1]<(segment+8)*480] if stage=='partner' else []
                        deltas.append(self.conditions(plans[ex['beat']//4],l.draft_features(partner,role,segment).to(self.model.device),role,goal))
                    values=er.batch_category_losses(self.model,chunk,deltas)
                    rows.extend(dict(group=w['group'],kind=ex['kind'],losses={k:float(v) for k,v in value.items()}) for ex,value in zip(chunk,values))
            scores[role]=er.aggregate(rows)
            print('Validated',stage,role,flush=True)
        return scores

    def audit(self,reference=None):
        """Evaluate exact current weights before any optimizer step."""
        if self.state['updates']:raise ValueError('Initial audit requires a fresh run with no optimizer updates')
        rng=rng_state();current=copy.deepcopy(self.bundle())
        try:
            self.status('Auditing current phrase prediction on 12 composition groups per role')
            phrase=self.validate('phrases')
            self.status('Auditing current partner-conditioned prediction')
            partner=self.validate('partner')
            report=dict(no_optimizer_updates=True,initial_sha256=sha(self.out/'initial.pt'),
                init_source_sha256=self.config['init_sha256'],current_phrase=phrase,current_partner=partner,
                phrase_macro_event=macro(phrase),partner_macro_event=macro(partner))
            if reference is not None:
                saved=safe_load(reference)
                if saved['base_sha256']!=self.base_hash or saved['roles']!=l.ROLES:raise ValueError('Audit reference incompatible')
                self.restore(saved);self.status('Auditing listener-preferred reference on the same compositions')
                reference_scores=self.validate('phrases')
                report.update(reference_sha256=sha(reference),reference_phrase=reference_scores,
                    reference_macro_event=macro(reference_scores),current_eligible_vs_reference=eligible(phrase,reference_scores))
            assert self.adapters.base_hash()==self.base_hash
            write_json(self.out/'initial_audit.json',report)
            print('AUDIT COMPLETE',json.dumps({k:v for k,v in report.items() if k.endswith('macro_event') or k=='current_eligible_vs_reference'}),flush=True)
        finally:self.restore(current);self.clear_grads();restore_rng(rng);self.save()

    def prepare_drafts(self,block):
        """Stage-start weights stay frozen; anchors rotate across compositions."""
        stage_root=self.out/'partner_drafts';stage_root.mkdir(exist_ok=True)
        path=stage_root/f'block_{block:05d}.pt'
        frozen=safe_load(self.out/'partner_start.pt')
        expected=sha(self.out/'partner_start.pt')
        if path.exists():
            saved=safe_load(path)
            if saved['source_sha256']!=expected:raise ValueError('Draft cache belongs to a different policy')
            self.state['draft_block']=block;return saved
        current=copy.deepcopy(self.bundle());rng=rng_state()
        try:
            self.restore(frozen);self.planner.eval();self.conditions.eval();anchors={};drafts={}
            for role in l.ROLES:
                pool=self.groups['train'][role];keys=sorted(pool)
                random.Random(self.args.seed+block*37+l.ROLES[role]['id']).shuffle(keys)
                key=keys[block%len(keys)]
                active=[w for w in pool[key] if any(n[3]==l.ROLES[role]['id'] and 3840<=n[1]<11520 for n in w['notes'])]
                anchors[role]=min(active or pool[key],key=lambda w:w['id'])['id']
            unique=sorted(set(anchors.values()))
            for index,key in enumerate(unique):
                self.status(f'Frozen draft block {block}: {index+1}/{len(unique)}')
                w=self.lookup[key];goal=torch.tensor(w['style'],device=self.model.device)
                for segment in (8,16):
                    drafts[key,segment]=l.private_draft(self.model,self.adapters,self.agents,self.planner,self.conditions,
                        l.visible(w['notes'],segment),segment,6,w['available_roles'],goal,self.args.seed+block*100000+index*31)
                if self.stop:raise InterruptedError('Stop requested during draft generation')
            saved=dict(anchors=anchors,drafts=drafts,source_sha256=expected,block=block)
            atomic(path,saved);self.state['draft_block']=block;return saved
        finally:self.restore(current);restore_rng(rng)

    def update(self,stage,step,drafts):
        self.optimizer.zero_grad(set_to_none=True);rng=random.Random(self.args.seed+step+100000*self.state['stage_index'])
        warmup=stage=='partner' and step<self.args.partner_warmup
        self.planner.requires_grad_(not warmup)
        total=0.;count=0;largest=0;per_role={};family_losses={};self.planner.train();self.conditions.train()
        for family in l.FAMILIES:
            self.adapters.activate(family,not warmup);items=[];roles=[r for r,s in l.ROLES.items() if s['family']==family];value=0.
            for role in roles:
                per_role[role]=0
                for phrase in range(self.args.phrases_per_role):
                    generated=stage=='partner' and (step+l.ROLES[role]['id']+phrase)%4==0
                    if generated:w=self.lookup[drafts['anchors'][role]]
                    else:
                        group=rng.choice(sorted(self.groups['train'][role]));w=rng.choice(self.groups['train'][role][group])
                    entries=self.teachers(w,role)
                    if not entries:continue
                    goal=torch.tensor(w['style'],device=self.model.device)
                    plans=l.plan_sequence(self.planner,w['notes'],6,w['available_roles'],goal)
                    target=torch.stack([l.description(w['notes'],bar,6,w['available_roles']) for bar in range(6)]).to(self.model.device)
                    normalization=len(l.FAMILIES)*len(roles)*self.args.phrases_per_role;weight=w['role_weights'][role]/normalization
                    ploss=.1*F.mse_loss(plans[2:],target[2:])*weight
                    if ploss.requires_grad:ploss.backward()
                    value+=float(ploss.detach());plans=plans.detach()
                    for ex,weights in zip(entries,l.category_weights(entries)):
                        segment=ex['beat']//8*8
                        announced=drafts['drafts'][w['id'],segment] if generated else [n for n in w['notes'] if segment*480<=n[1]<(segment+8)*480]
                        partner=[n for n in announced if n[3]!=l.ROLES[role]['id']] if stage=='partner' else []
                        items.append((ex,plans[ex['beat']//4],l.draft_features(partner,role,segment),role,goal,{k:v*weight for k,v in weights.items()}))
                    per_role[role]+=len(entries)
            for chunk in microchunks(items,self.state['physical_capacity'],self.state['token_budget']):
                largest=max(largest,len(chunk))
                deltas=[self.conditions(row[1],row[2].to(self.model.device),row[3],row[4]) for row in chunk]
                values=er.batch_category_losses(self.model,[row[0] for row in chunk],deltas,training=True)
                loss=sum(sum(v[k]*weight for k,weight in row[5].items()) for v,row in zip(values,chunk))
                if not torch.isfinite(loss):raise ValueError('Nonfinite supervised loss')
                loss.backward();value+=float(loss.detach());count+=len(chunk)
            family_losses[family]=value;total+=value
        if not count:raise ValueError('No supported target decisions')
        norm=torch.nn.utils.clip_grad_norm_(self.params,1.,error_if_nonfinite=True);self.optimizer.step()
        return dict(stage=stage,step=step+1,loss=total,examples=count,role_examples=per_role,family_losses=family_losses,
                    adaptation_mode='partner_conditioning_warmup' if warmup else 'joint',
                    physical_microbatch=largest,gradient_norm=float(norm),peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)

    def train(self):
        if self.state['complete']:return
        deadline=self.args.hours*3600
        while self.state['stage_index']<2:
            stage=['phrases','partner'][self.state['stage_index']];self.state['stage']=stage
            if not self.state['stage_ready']:
                atomic(self.out/(stage+'_start.pt'),self.bundle())
                audit_path=self.out/'initial_audit.json';audit=json.loads(audit_path.read_text()) if audit_path.exists() else None
                self.status('Baseline validation: '+stage)
                if audit and not self.state['updates'] and audit['initial_sha256']==sha(self.out/'initial.pt'):
                    baseline=audit['current_partner' if stage=='partner' else 'current_phrase']
                else:baseline=self.validate(stage)
                self.state.update(initial=baseline,best=macro(baseline),selected_step=0,stage_ready=True)
                if stage=='partner' and self.args.preserve_phrase:
                    self.status('Saving phrase-ability baseline before partner adaptation')
                    if audit and not self.state['updates'] and audit['initial_sha256']==sha(self.out/'initial.pt'):
                        self.state['phrase_guard_baseline']=audit['current_phrase']
                    else:self.state['phrase_guard_baseline']=self.validate('phrases')
                atomic(self.out/(stage+'.pt'),self.bundle());self.optimizer=self.make_optimizer();self.save()
            drafts=None
            for step in range(self.state['step'],self.args.steps):
                if self.stop or self.elapsed()>=deadline:
                    self.status('Stopped at optimizer boundary; resume is available');self.save();return
                if stage=='partner':drafts=self.prepare_drafts(step//self.args.draft_refresh) if drafts is None or self.state['draft_block']!=step//self.args.draft_refresh else drafts
                self.status('Training: '+stage);tick=time.monotonic();before_rng=rng_state();retry=True
                while retry:
                    retry=False
                    try:row=self.update(stage,step,drafts)
                    except torch.cuda.OutOfMemoryError:
                        self.clear_grads();gc.collect();torch.cuda.empty_cache();restore_rng(before_rng)
                        if self.state['physical_capacity']<=4:
                            self.save();raise RuntimeError('OOM even at batch <=4. Resume with smaller padded-token budget.')
                        self.state['physical_capacity']=max(4,self.state['physical_capacity']//2)
                        self.state['token_budget']=max(2048,self.state['token_budget']//2)
                        self.status('OOM: cleared partial gradients, retrying complete optimizer update');retry=True
                row['seconds']=time.monotonic()-tick;self.state['step']=step+1;self.state['updates'].append(row)
                if stage=='partner' and step+1==self.args.partner_warmup:
                    start=safe_load(self.out/'partner_start.pt')
                    current_adapters=self.adapters.state()
                    assert all(torch.equal(v,current_adapters[k]) for k,v in start['adapters'].items())
                    assert all(torch.equal(v,self.planner.state_dict()[k].cpu()) for k,v in start['planner'].items())
                    changed=sum(not torch.equal(v,self.conditions.state_dict()[k].cpu()) for k,v in start['conditions'].items())
                    assert changed>0
                    write_json(self.out/'warmup_integrity.json',dict(passed=True,updates=step+1,
                        adapters_unchanged=True,planner_unchanged=True,changed_condition_tensors=changed))
                print(json.dumps({k:row[k] for k in ['stage','step','loss','examples','physical_microbatch','seconds']}),flush=True)
                self.status('Training: '+stage)
                if (step+1)%self.args.validation_interval==0 or step+1==self.args.steps:
                    self.status('Held-out validation: '+stage);scores=self.validate(stage);passed=eligible(scores,self.state['initial'])
                    record=dict(stage=stage,step=step+1,scores=scores,passed=passed)
                    if stage=='partner' and self.args.preserve_phrase:
                        self.status('Checking preservation of phrase prediction')
                        guard=self.validate('phrases');guard_passed=no_role_regression(guard,self.state['phrase_guard_baseline'])
                        passed=passed and guard_passed
                        record.update(passed=passed,phrase_guard=guard,phrase_guard_passed=guard_passed)
                    self.state['validations'].append(record)
                    if passed and macro(scores)<self.state['best']:
                        self.state.update(best=macro(scores),selected_step=step+1,selected_mode='planned' if stage=='phrases' else 'revised')
                        atomic(self.out/(stage+'.pt'),self.bundle());print('Selected',stage,step+1,flush=True)
                    self.save()
                elif (step+1)%self.args.save_interval==0:self.save()
            atomic(self.out/(stage+'_candidate.pt'),self.bundle('planned' if stage=='phrases' else 'revised'))
            self.state['selections'][stage]=dict(step=self.state['selected_step'],passed=self.state['selected_step']>0)
            chosen=safe_load(self.out/(stage+'.pt'));self.restore(chosen);self.state['selected_mode']=chosen['composition_mode']
            self.state.update(stage_index=self.state['stage_index']+1,step=0,stage_ready=False,draft_block=None)
            self.optimizer=self.make_optimizer();self.save()
        assert self.adapters.base_hash()==self.base_hash
        atomic(self.out/'selected.pt',self.bundle());self.state['complete']=True
        self.status('Complete: checkpoint selection done; listening preference unknown');self.save()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['verify','benchmark','train','smoke','audit'],default='train')
    p.add_argument('--output',type=Path,default=ROOT/'runs/h100_orchestra_v1')
    p.add_argument('--init',type=Path,default=ROOT/'checkpoints/laptop_phrases.pt')
    p.add_argument('--steps',type=int,default=512);p.add_argument('--phrases-per-role',type=int,default=8)
    p.add_argument('--microbatch',type=int,default=0,help='0 benchmarks 16..512 and selects throughput/headroom')
    p.add_argument('--token-budget',type=int,default=131072);p.add_argument('--learning-rate',type=float,default=1e-4)
    p.add_argument('--hours',type=float,default=6);p.add_argument('--cpu-threads',type=int,default=8)
    p.add_argument('--cache-entries',type=int,default=128);p.add_argument('--validation-groups',type=int,default=12)
    p.add_argument('--validation-interval',type=int,default=64);p.add_argument('--save-interval',type=int,default=8)
    p.add_argument('--draft-refresh',type=int,default=128);p.add_argument('--seed',type=int,default=9931001)
    p.add_argument('--resume',action='store_true');p.add_argument('--allow-non-h100',action='store_true')
    p.add_argument('--gradient-checkpointing',action='store_true')
    p.add_argument('--start-stage',choices=['phrases','partner'],default='phrases')
    p.add_argument('--partner-warmup',type=int,default=0)
    p.add_argument('--audit-reference',type=Path)
    p.add_argument('--preserve-phrase',action='store_true',help='Require per-role phrase event/attribute loss to stay within 2%% during partner training')
    args=p.parse_args()
    if not 1<=args.steps<=10000 or not 1<=args.phrases_per_role<=32 or not 0<=args.microbatch<=1024 or args.token_budget<1024 or args.hours<=0 or not 1<=args.validation_groups<=12 or min(args.validation_interval,args.save_interval,args.draft_refresh)<1:p.error('Invalid training settings')
    if args.mode=='train' and args.validation_groups<12:p.error('Full training requires all 12 validation groups per role')
    if not 0<=args.partner_warmup<=args.steps:p.error('Warmup must fit within the bounded stage')
    runner=Runner(args)
    def stop(signum,frame):runner.stop=True;print('Stop requested: checkpointing at the next complete update',flush=True)
    if hasattr(signal,'SIGTERM'):signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    try:
        if args.mode in ['verify','smoke']:runner.verify()
        if args.mode=='benchmark' or args.mode=='train' and not args.resume and not args.microbatch:runner.calibrate()
        if args.mode=='audit':runner.audit(args.audit_reference)
        if args.mode=='train':
            if not (runner.out/'verification.json').exists():runner.verify()
            runner.train()
        if args.mode=='smoke':
            before=copy.deepcopy(runner.bundle());row=runner.update('phrases',0,None)
            runner.state['updates'].append(row);runner.state['step']=1;runner.save();checkpoint=safe_load(runner.out/'progress.pt')
            expected=copy.deepcopy(runner.adapters.state());runner.restore(before);runner.restore(checkpoint)
            assert all(torch.equal(v,runner.adapters.state()[k]) for k,v in expected.items())
            opts=runner.make_optimizer();opts.load_state_dict(checkpoint['optimizer']);restore_rng(checkpoint['rng'])
            runner.adapters.activate('organ');runner.planner.eval();runner.conditions.eval()
            w=fixed_windows(runner.groups['train'],'pipe_organ',1)[0]
            drafts=l.private_draft(runner.model,runner.adapters,runner.agents,runner.planner,runner.conditions,
                l.visible(w['notes'],16),16,6,['pipe_organ'],torch.tensor(w['style'],device=runner.model.device),9912345)
            assert all(n[1]+n[2]<=11520 for n in drafts)
            runner.status('Smoke passed: optimizer update, checkpoint/optimizer/RNG load, and end-boundary draft')
            write_json(runner.out/'smoke_result.json',dict(passed=True,update_examples=row['examples'],
                all_role_example_counts=row['role_examples'],optimizer_reloaded=True,weights_reloaded=True,
                rng_reloaded=True,end_boundary_draft=True,h100_executed='H100' in torch.cuda.get_device_name()))
    except (KeyboardInterrupt,InterruptedError):runner.status('Stopped; resume from the last complete saved update')
    except BaseException as error:
        runner.state['failure']=str(error);runner.status('Failed: '+str(error));raise


if __name__=='__main__':main()
