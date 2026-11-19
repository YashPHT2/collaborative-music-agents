"""Bounded, resumable two-stage ensemble repair; no backbone training or PPO."""
import argparse
import copy
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import composition_data as cd
import ensemble_repair as er
import offline_ensemble as oe
import supervised_ensemble as se
import note_trio as nt
from train_offline_ensemble import atomic

OUT = nt.ROOT/'outputs'/'ensemble_repair_v1'
SOURCE = nt.ROOT/'outputs'/'offline_ensemble_v1'/'planned.pt'


def write_json(path, value):
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    tmp.replace(path)


def corpus():
    records=cd.build()
    import ensemble_data as ed
    source_roles={rec['song']:[r for r,programs in ed.PROGRAMS.items() if any(n['program'] in programs for n in rec['raw'])] for rec in records}
    windows = oe.candidates(records, 8)
    groups = {split: {r: {} for r in er.ROLES} for split in ('train', 'validation', 'test')}
    audit = {split: {r: Counter() for r in er.ROLES} for split in groups}
    for w in windows:
        w['source_roles']=source_roles[w['song']]
        for role in er.ROLES:
            audit[w['split']][role].update(er.beat_kind(w, role, b) for b in range(32))
            if role in w['available_roles']:
                groups[w['split']][role].setdefault(w['group'], []).append(w)
    # Equal composition contribution to profile quantiles; never use held-out data.
    unique = {}
    for w in windows:
        if w['split'] == 'train':
            unique.setdefault(w['group'], w)
    quantiles = {}
    for channel, role in enumerate(er.ROLES):
        representatives = [ws[0] for _, ws in sorted(groups['train'][role].items())]
        values = torch.stack([er.description(w['notes'], b) for w in representatives for b in range(8)])
        for i in (12+channel, 15+channel, 18+channel):
            # Register/velocity are undefined in silent bars; density retains rests.
            observations = values[:, i] if i < 15 else values[:, i][values[:, 12+channel] > 0]
            quantiles[str(i)] = torch.quantile(observations, torch.tensor([.35, .65])).tolist()
    return windows, groups, audit, quantiles


def validation_windows(groups, role):
    rng = random.Random(972001+er.ROLES.index(role))
    keys = sorted(groups[role]); rng.shuffle(keys)
    return [rng.choice(groups[role][g]) for g in keys[:12]]


def supported_categories(ex):
    mask = ex['support'].any(-1)
    keys = [name for name, col in [('pitch',5),('onset',2),('duration',7),('velocity',6)] if mask[:,col].any()]
    if mask[0,0]:keys.append('first_event')
    if mask[1:,0].any():keys.append('additional_event')
    return keys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--steps', type=int, default=512)
    parser.add_argument('--hours', type=float, default=3)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--microbatch', type=int, default=8)
    parser.add_argument('--phrases-per-role', type=int, default=2)
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--phrase-focus',action='store_true',help='Four-bar phrases, melodic curation and contour loss')
    args = parser.parse_args()
    if not 1 <= args.steps <= 512 or not 0 < args.hours <= 3:
        parser.error('steps must be 1..512 and hours must be >0 and <=3')
    if args.microbatch not in (1,2,4,8,16,32,64,128) or not 1<=args.phrases_per_role<=16:
        parser.error('microbatch must be 1,2,4,8,16,32,64,128; phrases-per-role must be 1..16')
    root = args.output; root.mkdir(parents=True, exist_ok=True)
    if (root/'progress.pt').exists() and not args.resume:
        raise ValueError('Use --resume or an empty output directory')
    windows, groups, audit, quantiles = corpus()
    write_json(root/'data_audit.json', dict(beat_labels=audit, independent_compositions={s:{r:len(g) for r,g in rs.items()} for s,rs in groups.items()},
              missing_definition='No matching source program; excluded_low_support means a labeled part has fewer than eight projected notes in this window. Both are masked, never silence.',
              profile_quantiles=quantiles, profile_quantiles_split='train',
              reference_participation=[dict(group=w['group'], metrics=er.participation(w['notes'])) for w in windows if w['split']=='validation']))
    if args.audit_only:
        print(json.dumps(audit, indent=2));return
    if any(len(groups['validation'][r])<12 for r in er.ROLES):
        raise ValueError('Need twelve independent validation compositions for each role')
    config = dict(steps=args.steps, hours=args.hours, source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                  corpus_sha256=hashlib.sha256((cd.ROOT/'corpus.json.gz').read_bytes()).hexdigest(), version=3,
                  loss_objective='masked_cross_entropy_without_exploration_floor')
    if args.phrase_focus:
        config['phrase_focus']=dict(beats=16,contour_weight=.15,curated_probability=.75)
        import phrase_curriculum as pc
        write_json(root/'phrase_curriculum.json',{r:dict(compositions=len(groups['train'][r]),
                    compositions_with_preferred_material=sum(any(pc.preferred(w,r) for w in ws) for ws in groups['train'][r].values())) for r in er.ROLES})
        focused_groups={r:[g for g,ws in groups['train'][r].items() if any(pc.preferred(w,r) for w in ws)] for r in er.ROLES}
    torch.set_num_threads(4);torch.manual_seed(972001)
    model = nt.load_model();adapters = se.InstrumentAdapters(model)
    source = torch.load(args.source, map_location='cpu', weights_only=False)
    adapters.load(source['adapters']);planner, conditions = er.upgrade(source, model.device)
    base_hash = adapters.base_hash()
    if base_hash != source['base_sha256']:
        raise ValueError('Backbone differs from the source checkpoint')
    agents = {r:oe.Musician().eval() for r in er.ROLES}
    params = [p for r in er.ROLES for p in adapters.params(r)]+list(planner.parameters())+list(conditions.parameters())
    optimizer = torch.optim.AdamW(params, lr=3e-5, weight_decay=.01)
    state = dict(stage=0, step=0, elapsed=0., started=False, logs=[], selections={}, complete=False)
    cache = {}
    def bundle():
        return dict(format_version=3, plan_size=48, adapters=adapters.state(), planner=planner.state_dict(), conditions=conditions.state_dict(),
                    constraints=dict(strings_cap=4,max_duration_ticks=3840), config=config, base_sha256=base_hash,
                    profile_quantiles=quantiles, selected_stage=state.get('selected_stage','initial'),
                    composition_mode=state.get('composition_mode','planned'), source=str(args.source.resolve()))
    if args.resume:
        saved = torch.load(root/'progress.pt', map_location='cpu', weights_only=False)
        old_config=saved['config']
        migration='loss_objective' not in old_config
        if {**old_config,'loss_objective':config['loss_objective']} != config:raise ValueError('Resume configuration differs')
        adapters.load(saved['adapters']);planner.load_state_dict(saved['planner']);conditions.load_state_dict(saved['conditions'])
        optimizer.load_state_dict(saved['optimizer']);state=saved['state'];cache=saved['cache']
        if migration:
            import shutil
            backup=root/'progress_before_objective_fix.pt'
            if not backup.exists():shutil.copy2(root/'progress.pt',backup)
            state['objective_migration']=dict(stage=state['stage'],step=state['step'],
                                            reason='Exploration-floor saturation suppressed gradients; validation still uses sampler probabilities')
            # Include the interval since the last checkpoint in the shared budget.
            state['elapsed']+=max(0,time.time()-(root/'progress.pt').stat().st_mtime)
        torch.set_rng_state(saved['rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
    old_batch=state.get('batch',dict(microbatch=1,phrases_per_role=1))
    new_batch=dict(microbatch=args.microbatch,phrases_per_role=args.phrases_per_role,
                   maximum_beat_examples_per_update=24*args.phrases_per_role)
    if old_batch!=new_batch:
        state.setdefault('batch_changes',[]).append(dict(stage=state['stage'],step=state['step'],previous=old_batch,current=new_batch))
        if args.resume:
            state['elapsed']+=max(0,time.time()-(root/'progress.pt').stat().st_mtime)
    state['batch']=new_batch
    print('Training batch:',new_batch,flush=True)
    started=time.monotonic();prior=state['elapsed']
    # Reserve 45 minutes of the shared wall-clock budget for listening/probes.
    training_limit=max(0,args.hours*3600- min(2700,args.hours*3600*.25))
    def save():
        state['elapsed']=prior+time.monotonic()-started
        atomic(root/'progress.pt',{**bundle(), 'optimizer':optimizer.state_dict(), 'state':state, 'cache':cache,
                                  'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()})
        write_json(root/'training_log.json',state)
    def restore(saved):
        adapters.load(saved['adapters']);planner.load_state_dict(saved['planner']);conditions.load_state_dict(saved['conditions'])
    frozen_stage={}
    def generated(w, stage):
        key=w['fingerprint']
        if key not in cache:
            current=copy.deepcopy(bundle())
            if stage not in frozen_stage:
                frozen_stage.clear()
                frozen_stage[stage]=torch.load(root/f'{stage}_start.pt',map_location='cpu',weights_only=False)
            frozen=frozen_stage[stage]
            restore(frozen);planner.eval();conditions.eval()
            seed=972100+int(key[:6],16)
            ep=er.compose(model,adapters,agents,planner,conditions,8,seed,'planned',prefix=nt.visible(w['notes'],8))
            cache[key]=ep['notes']
            restore(current)
            # Drafts depend only on frozen stage weights, fingerprint and seed.
            # Retain all visited windows for the stage; these are small CPU note
            # lists, not CUDA tensors. Reuse changes no sampling distribution.
        return cache[key]
    @torch.no_grad()
    def validate(stage):
        planner.eval();conditions.eval();scores={}
        for role in er.ROLES:
            adapters.activate(role);rows=[]
            for j,w in enumerate(validation_windows(groups['validation'],role)):
                plans=er.plan_sequence(planner,w['notes'],8)
                examples=[]
                for beat in range(32):
                    with oe.limits():
                        ex=se.teacher_example(model,role,{**w,'beat':beat,'sampler_type':oe.Sampler})
                    if ex is not None:examples.append((beat,ex))
                # Same 32 held-out beats and exact masks, evaluated in parallel.
                for offset in range(0,len(examples),16):
                    chunk=examples[offset:offset+16];deltas=[]
                    for beat,ex in chunk:
                        start=beat//8*3840
                        partners=[n for n in w['notes'] if n[3]!=er.ROLES.index(role) and start<=n[1]<start+3840] if stage=='revision' else []
                        deltas.append(conditions(plans[beat//4],oe.draft_features(partners,role,start).to(model.device),role))
                    losses=er.batch_category_losses(model,[ex for _,ex in chunk],deltas)
                    for (beat,ex),values in zip(chunk,losses):
                        rows.append(dict(group=w['group'],kind=er.beat_kind(w,role,beat),losses={k:float(v) for k,v in values.items()}))
            scores[role]=er.aggregate(rows)
        return scores
    if not (root/'initial.pt').exists():atomic(root/'initial.pt',bundle())
    save()
    try:
        for stage_index in range(state['stage'],2):
            stage=['participation','revision'][stage_index]
            state['stage']=stage_index
            if not state['started']:
                cache={};optimizer=torch.optim.AdamW(params,lr=3e-5,weight_decay=.01)
                state['stage_elapsed_start']=prior+time.monotonic()-started
                atomic(root/f'{stage}_start.pt',bundle())
                print('Validating stage start:',stage,flush=True)
                state['initial']=validate(stage);state['best']=sum(s['event'] for s in state['initial'].values())/3
                state['selected_step']=0;state['started']=True
                atomic(root/f'{stage}.pt',bundle());save()
            for step in range(state['step'],args.steps):
                elapsed=prior+time.monotonic()-started
                if elapsed>=training_limit:
                    state['budget_stopped']=True;break
                if elapsed-state.get('stage_elapsed_start',0.)>=training_limit/2:
                    state.setdefault('stage_budget_stops',[]).append(stage);break
                rng=random.Random(972001+stage_index*100000+step)
                update_started=time.monotonic();torch.cuda.reset_peak_memory_stats()
                optimizer.zero_grad();total=0.;used=0;timings=dict(preparation=0.,gpu_training=0.)
                for role in er.ROLES:
                    prepare_started=time.monotonic()
                    batches=[]
                    for phrase_index in range(args.phrases_per_role):
                        pool=focused_groups[role] if args.phrase_focus and focused_groups[role] and rng.random()<.75 else sorted(groups['train'][role])
                        group=rng.choice(pool)
                        w=pc.choose_window(groups['train'][role][group],role,rng) if args.phrase_focus else rng.choice(groups['train'][role][group])
                        phrase_beats=16 if args.phrase_focus else 8
                        start=er.phrase_choice(w,role,rng)//phrase_beats*phrase_beats
                        probability=(.15 if args.phrase_focus else .5)*min(1,step/max(1,args.steps//2))
                        use_generated=rng.random()<probability
                        use_draft=stage=='revision' and rng.random()<.5
                        synthetic=generated(w,stage) if use_generated or use_draft else None
                        adapters.activate(role,training=True);planner.train();conditions.train()
                        plans=er.plan_sequence(planner,synthetic if use_generated else w['notes'],8)
                        target=torch.stack([er.description(w['notes'],b) for b in range(8)]).to(model.device)
                        ploss=.1*F.mse_loss(plans,target)/(3*args.phrases_per_role)
                        ploss.backward();total+=float(ploss.detach());plans=plans.detach()
                        draft=(synthetic if use_draft else w['notes']) if stage=='revision' else []
                        entries=[];counts={k:Counter() for k in er.CATEGORIES}
                        for beat in range(start,start+phrase_beats):
                            with oe.limits():ex=se.teacher_example(model,role,{**w,'beat':beat,'sampler_type':oe.Sampler})
                            if ex is None:continue
                            kind=er.beat_kind(w,role,beat);active=kind=='onset';keys=supported_categories(ex)
                            entries.append((beat,active,ex,keys))
                            for k in keys:counts[k][active]+=1
                        available=sum(bool(c) for c in counts.values())
                        for beat,active,ex,keys in entries:
                            segment=beat//8*3840
                            partners=[n for n in draft if n[3]!=er.ROLES.index(role) and segment<=n[1]<segment+3840]
                            weights={k:1/(counts[k][active]*len(counts[k])*available*3*args.phrases_per_role) for k in keys}
                            batches.append((ex,plans[beat//4],oe.draft_features(partners,role,segment).to(model.device),weights,phrase_index,beat))
                    # Similar context lengths share a physical CUDA microbatch.
                    batches.sort(key=(lambda b:(b[4],b[5])) if args.phrase_focus else (lambda b:len(b[0]['x'])))
                    timings['preparation']+=time.monotonic()-prepare_started
                    gpu_started=time.monotonic()
                    for offset in range(0,len(batches),args.microbatch):
                        chunk=batches[offset:offset+args.microbatch]
                        deltas=[conditions(b[1],b[2],role) for b in chunk]
                        result=er.batch_category_losses(model,[b[0] for b in chunk],deltas,training=True,return_pitch=args.phrase_focus)
                        losses,pitch_stats=result if args.phrase_focus else (result,None)
                        loss=sum(sum(values[k]*weight for k,weight in row[3].items()) for values,row in zip(losses,chunk))
                        if args.phrase_focus:
                            contour=pc.contour_loss(pitch_stats,[b[4] for b in chunk])
                            if contour is not None:loss=loss+.15*contour/(3*max(1,len(batches)/args.microbatch))
                        if not torch.isfinite(loss):raise ValueError('Nonfinite repair loss')
                        loss.backward();total+=float(loss.detach());used+=len(chunk)
                    timings['gpu_training']+=time.monotonic()-gpu_started
                if not used:raise ValueError('No supported phrase decisions')
                torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step()
                state['step']=step+1
                state['last_update']=dict(stage=stage,step=step+1,loss=total,examples=used,
                                          seconds=time.monotonic()-update_started,
                                          phase_seconds=timings,
                                          peak_cuda_reserved_mib=torch.cuda.max_memory_reserved()/1024**2)
                print(f'{stage} {step+1}/{args.steps}: loss={total:.4f}',flush=True)
                validation_interval=max(8,64//args.phrases_per_role)
                checkpoint_interval=max(1,16//args.phrases_per_role)
                if (step+1)%validation_interval==0 or step+1==args.steps:
                    scores=validate(stage);eligible=er.eligible(scores,state['initial'])
                    mean=sum(s['event'] for s in scores.values())/3
                    state['logs'].append(dict(stage=stage,step=step+1,loss=total,validation=scores,eligible=eligible))
                    if eligible and mean<state['best']:
                        state['best']=mean;state['selected_step']=step+1
                        state['selected_stage']=stage;state['composition_mode']='revised' if stage=='revision' else 'planned'
                        atomic(root/f'{stage}.pt',bundle())
                    save()
                elif (step+1)%checkpoint_interval==0:save()
            if args.phrase_focus:atomic(root/f'{stage}_candidate.pt',bundle())
            chosen=torch.load(root/f'{stage}.pt',map_location='cpu',weights_only=False)
            restore(chosen)
            state['selected_stage']=chosen['selected_stage'];state['composition_mode']=chosen['composition_mode']
            state['selections'][stage]=dict(step=state['selected_step'],passed=state['selected_step']>0)
            state.update(stage=stage_index+1,step=0,started=False);save()
            if state.get('budget_stopped'):break
        if adapters.base_hash()!=base_hash:raise ValueError('Frozen backbone changed')
        state['complete']=True
        atomic(root/'latest.pt',bundle());save()
        write_json(root/'campaign_status.json',dict(stage='training_complete',selections=state['selections'],quality='not established'))
        print('Repair training complete:',state['selections'],flush=True)
    except BaseException:
        save();raise


if __name__=='__main__':main()
