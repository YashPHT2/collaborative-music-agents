"""Paired real-history control versus bounded anchored history augmentation."""
import collections,copy,gzip,json,math,random,time
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
import ensemble_repair as er
import rock_history_mix as mix
from train_rock_pilot import atomic,write,sha,chunks
g=mix.g;l=g.l;rock=mix.rock
ROOT=Path(__file__).resolve().parent

def macro(scores):
    return float(np.mean([np.mean([scores[r]['event'] for r,s in rock.ROLES.items() if s['family']==family]) for family in rock.FAMILIES]))
def objective(scores):return .25*macro(scores['startup'])+.375*macro(scores['real'])+.375*macro(scores['mixed'])
def eligible(scores,baseline):
    for task in baseline:
        for role in rock.ROLES:
            for metric in ['event','attributes']:
                now=scores[task][role][metric];old=baseline[task][role][metric]
                if now is None or old is None or not np.isfinite(now) or now>old*1.02:return False
    return objective(scores)<objective(baseline)

def weights_sha(saved):
    h=mix.hashlib.sha256()
    for component in ['adapters','planner','conditions']:
        for key,tensor in sorted(saved[component].items()):
            h.update((component+'/'+key).encode());h.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

@torch.no_grad()
def evaluation(output,source,model,adapters,planner,conditions,actors,goal,deadline,on_save):
    directory=output/'evaluation';directory.mkdir(exist_ok=True)
    models={'baseline':source}
    for name in ['control','mixed']:models[name]=torch.load(output/name/'selected.pt',map_location='cpu',weights_only=False)
    seen={};aliases={};artifacts={}
    for name,saved in models.items():
        fingerprint=weights_sha(saved)
        if fingerprint in seen:aliases[name]=seen[fingerprint];continue
        seen[fingerprint]=name;aliases[name]=name
        adapters.load(saved['adapters']);planner.load_state_dict(saved['planner']);conditions.load_state_dict(saved['conditions']);planner.eval();conditions.eval()
        for seed in [9976501,9976502,9976503]:
            if time.monotonic()>=deadline:break
            p=dict(prefix=[],goal=goal,bpm=120,seed=seed)
            episode=mix.diagnostic.compose(model,adapters,planner,conditions,actors,p,'unprompted')
            path=directory/f'{name}_{seed}.json';write(path,episode)
            artifacts[path.name]=dict(sha256=sha(path),model=name,seed=seed,weights_sha256=fingerprint,notes=len(episode['notes']))
            on_save();print(json.dumps(dict(event='evaluation_ready',model=name,seed=seed,notes=len(episode['notes']))),flush=True)
    result=dict(schema='rock-history-mix-evaluation-v1',seeds=[9976501,9976502,9976503],bars=6,bpm=120,reference_prefix=False,goal=goal,aliases=aliases,artifacts=artifacts,
        complete=len(artifacts)==3*len(seen),note='Identical selected weights share one generated artifact per seed; no attractive-sample selection.')
    write(directory/'manifest.json',result);on_save();return result

class Arm:
    def __init__(self,name,output,model,adapters,planner,conditions,source,windows,records,goal,deadline,on_save):
        self.name=name;self.out=output/name;self.out.mkdir(exist_ok=True);self.deadline=deadline;self.on_save=on_save
        if (self.out/'progress.pt').exists():raise ValueError('Existing arm preserved; do not overwrite an interrupted experiment')
        self.model=model;self.adapters=adapters;self.planner=planner;self.conditions=conditions;self.source=source
        self.base=source['base_sha256'];self.goal=torch.tensor(goal,device=model.device);self.windows=windows;self.records=records
        self.lookup={w['id']:w for w in windows};self.cache=collections.OrderedDict()
        self.groups={s:{r:collections.defaultdict(list) for r in rock.ROLES} for s in ['train','validation']}
        self.banks={s:{r:collections.defaultdict(list) for r in rock.ROLES} for s in ['train','validation']}
        for w in windows:
            if w['split'] not in self.groups:continue
            for role in w['available_roles']:
                self.groups[w['split']][role][w['group']].append(w)
                if w['id'] in records:self.banks[w['split']][role][w['group']].append(w)
        for role in rock.ROLES:
            if len(self.banks['train'][role])<4 or len(self.banks['validation'][role])<3:raise ValueError('History cache has insufficient role/group coverage: '+role)
        self.fixed={r:[min(self.banks['validation'][r][group],key=lambda w:w['id']) for group in sorted(self.banks['validation'][r])[:3]] for r in rock.ROLES}
        adapters.load(source['adapters']);planner.load_state_dict(source['planner']);conditions.load_state_dict(source['conditions'])
        for family in rock.FAMILIES:
            for p in adapters.params(family):p.data=p.data.float()
        self.params=[p for family in rock.FAMILIES for p in adapters.params(family)]+list(conditions.parameters())
        self.allparams=self.params+list(planner.parameters())
        self.optimizer=torch.optim.AdamW([dict(params=self.params,lr=2e-5),dict(params=planner.parameters(),lr=1e-5)],weight_decay=.01)
        self.config=dict(arm=name,updates=64,phrases_per_role=8,physical_capacity=64,token_budget=32768,adapter_condition_lr=2e-5,planner_lr=1e-5,
            startup_loss_weight=.25,continuation_loss_weight=.75,goal=goal,max_generated_phrase_fraction=.5 if name=='mixed' else 0.,
            schedule='Within half the batch, linearly ramp replacement probability from 0 to 1 over 16 updates.',
            history_policy='Frozen original step128; one generated bar following one real bar. Unsupported examples fall back to original real context.',
            init_step=128,init_sha256=source['history_mix_source_sha256'],data_sha256=source['history_mix_data_sha256'],teacher_targets='Original source targets; generated notes are context only.',rng_seed=9976400)
        self.state=dict(stage='history_mix_'+name,step=0,total=64,selected_step=0,complete=False,init_step=128,updates=[],validations=[])

    def examples(self,w,role,augmented):
        key=(w['id'],role,augmented)
        if key not in self.cache:
            self.cache[key]=mix.examples(self.model,w,role,self.records[w['id']] if augmented else None)
            if len(self.cache)>96:self.cache.popitem(last=False)
        self.cache.move_to_end(key);return self.cache[key]

    def bundle(self):
        return dict(format_version='rock-pilot-v1',roles=rock.ROLES,families=rock.FAMILIES,adapters=self.adapters.state(),
            planner={k:v.detach().cpu().clone() for k,v in self.planner.state_dict().items()},conditions={k:v.detach().cpu().clone() for k,v in self.conditions.state_dict().items()},
            base_sha256=self.base,config=self.config,state=copy.deepcopy(self.state),optimizer=self.optimizer.state_dict(),
            rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),python=random.getstate(),numpy=np.random.get_state()))
    def save(self):
        atomic(self.out/'progress.pt',self.bundle());write(self.out/'training_log.json',self.state);self.on_save()

    @torch.no_grad()
    def validate(self):
        self.planner.eval();self.conditions.eval();result={k:{} for k in ['startup','real','mixed']};coverage={}
        for role,s in rock.ROLES.items():
            self.adapters.activate(s['family']);rows={k:[] for k in result};support=collections.Counter()
            for w in self.fixed[role]:
                cold,real,_=self.examples(w,role,False);_,augmented,stats=self.examples(w,role,True)
                support.update({k:stats[k] for k in ['requested','used','fallback']})
                original_plans=l.plan_sequence(self.planner,w['notes'],6,w['available_roles'],self.goal)
                mixed_plans=mix.plan_sequence(self.planner,w,self.goal,self.records[w['id']])
                for task,exs,plans in [('startup',cold,original_plans),('real',real,original_plans),('mixed',augmented,mixed_plans)]:
                    for chunk in chunks([(ex,None) for ex in exs],32):
                        examples=[x[0] for x in chunk];deltas=[self.conditions((plans if task!='mixed' or ex.get('history_mixed') else original_plans)[ex['beat']//4],torch.zeros(1,8,device=self.model.device),role,self.goal) for ex in examples]
                        values=er.batch_category_losses(self.model,examples,deltas)
                        rows[task].extend(dict(group=w['group'],kind=ex['kind'],losses={k:float(v) for k,v in value.items()}) for ex,value in zip(examples,values))
            for task in result:
                result[task][role]=er.aggregate(rows[task])
                if result[task][role]['event'] is None or result[task][role]['attributes'] is None:raise ValueError('Missing validation support '+task+'/'+role)
            coverage[role]=dict(support)
            print(json.dumps(dict(event='validated_role',arm=self.name,step=self.state['step'],role=role,real_event=result['real'][role]['event'],mixed_event=result['mixed'][role]['event'])),flush=True)
        self.state['validation_history_support']=coverage
        return result

    def update(self,step):
        import ensemble_repair as er
        self.optimizer.zero_grad(set_to_none=True);self.planner.train();self.conditions.train()
        # The same sample plan in both arms; mixture draws never alter sample RNG.
        rng=random.Random(9976400+step);total=0.;count=0;role_stats={};sample_ids=[];largest=0
        for family in rock.FAMILIES:
            self.adapters.activate(family,training=True);roles=[r for r,s in rock.ROLES.items() if s['family']==family];items=[]
            for role in roles:
                stats=collections.Counter()
                for phrase in range(8):
                    pool=self.groups['train'][role] if phrase<4 else self.banks['train'][role]
                    group=rng.choice(sorted(pool));w=rng.choice(sorted(pool[group],key=lambda w:w['id']));sample_ids.append([role,w['id'],phrase])
                    coin=random.Random(9976400+step*10000+rock.ROLES[role]['id']*101+phrase).random()
                    augmented=self.name=='mixed' and phrase>=4 and coin<min(1.,(step+1)/16)
                    cold,continuation,coverage=self.examples(w,role,augmented)
                    stats.update({k:coverage[k] for k in ['requested','used','changed','fallback']});stats['phrases']+=1;stats['augmented_phrases']+=augmented
                    stats['startup_examples']+=len(cold);stats['continuation_examples']+=len(continuation)
                    real_plans=l.plan_sequence(self.planner,w['notes'],6,w['available_roles'],self.goal)
                    plans=mix.plan_sequence(self.planner,w,self.goal,self.records[w['id']]) if augmented else real_plans
                    target=torch.stack([l.description(w['notes'],b,6,w['available_roles']) for b in range(6)]).to(self.model.device)
                    norm=len(rock.FAMILIES)*len(roles)*8
                    ploss=.1*(.5*F.mse_loss(plans,target)+.5*F.mse_loss(real_plans,target))/norm
                    ploss.backward();total+=float(ploss.detach());plans=plans.detach();real_plans=real_plans.detach()
                    for exs,task_weight in [(cold,.25),(continuation,.75)]:
                        for ex,weights in zip(exs,l.category_weights(exs)):
                            # Per-example fallback uses the matching original plan.
                            plan=(plans if ex.get('history_mixed') else real_plans)[ex['beat']//4]
                            items.append((ex,plan,role,{k:v*task_weight/norm for k,v in weights.items()}))
                role_stats[role]=dict(stats)
            for chunk in chunks(items,64):
                examples=[x[0] for x in chunk];deltas=[self.conditions(x[1],torch.zeros(1,8,device=self.model.device),x[2],self.goal) for x in chunk]
                values=er.batch_category_losses(self.model,examples,deltas,training=True)
                loss=sum(sum(v[k]*weight for k,weight in row[3].items()) for v,row in zip(values,chunk))
                if not torch.isfinite(loss):raise ValueError('Nonfinite mixed-history loss')
                loss.backward();total+=float(loss.detach());count+=len(examples);largest=max(largest,len(chunk))
        assert not any(p.grad is not None for n,p in self.model.named_parameters() if 'lora_' not in n)
        norm=torch.nn.utils.clip_grad_norm_(self.allparams,1.,error_if_nonfinite=True)
        factor=(step+1)/4 if step<4 else .5*(1+math.cos(math.pi*(step-4)/60))
        for group,lr in zip(self.optimizer.param_groups,[2e-5,1e-5]):group['lr']=lr*factor
        self.optimizer.step()
        return dict(step=step+1,loss=total,examples=count,role_history=role_stats,physical_microbatch=largest,gradient_norm=float(norm),sample_plan_sha256=mix.hashlib.sha256(json.dumps(sample_ids).encode()).hexdigest())

    def train(self):
        # No OOM retry on partial gradients: save failure and preserve last complete update.
        print(json.dumps(dict(event='baseline_validation',arm=self.name)),flush=True)
        baseline=self.validate();best=objective(baseline);self.state['initial_validation']=baseline
        atomic(self.out/'selected.pt',self.bundle());self.save()
        for step in range(64):
            if time.monotonic()>=self.deadline:break
            tick=time.monotonic();row=self.update(step);row['seconds']=time.monotonic()-tick
            self.state['step']=step+1;self.state['updates'].append(row);print(json.dumps(dict(arm=self.name,**row)),flush=True)
            if (step+1)%16==0:
                scores=self.validate();passed=eligible(scores,baseline);value=objective(scores)
                self.state['validations'].append(dict(step=step+1,scores=scores,passed=passed,objective=value))
                if passed and value<best:self.state['selected_step']=step+1;best=value;atomic(self.out/'selected.pt',self.bundle())
            if (step+1)%8==0:self.save()
        self.state['complete']=self.state['step']==64
        assert self.adapters.base_hash()==self.base
        self.state['frozen_backbone_unchanged']=True;self.save();return copy.deepcopy(self.state)

def run(output,source_path,on_save=lambda:None):
    rock.configure();torch.set_num_threads(8);torch.manual_seed(9976400);random.seed(9976400);np.random.seed(9976400)
    started=time.monotonic();deadline=started+7200;output=Path(output);output.mkdir(exist_ok=True)
    source_path=Path(source_path);source_sha=sha(source_path);source=torch.load(source_path,map_location='cpu',weights_only=False)
    if source['state']['step']!=128 or source['format_version']!='rock-pilot-v1':raise ValueError('Expected original step128')
    source.pop('optimizer',None);source.pop('rng',None)
    data=ROOT/'outputs/rock_training_v1';manifest=json.loads((data/'rock_manifest.json').read_text());assert sha(data/'rock_examples.json.gz')==manifest['examples_sha256']
    source['history_mix_source_sha256']=source_sha;source['history_mix_data_sha256']=manifest['examples_sha256']
    with gzip.open(data/'rock_examples.json.gz','rt') as f:windows=json.load(f)
    goal=np.mean([l.style([n for n in w['notes'] if n[3]!=3],w['available_roles']).numpy() for w in windows if w['split']=='train'],axis=0).tolist()
    model=g.nt.load_model();adapters=l.Adapters(model);assert adapters.base_hash()==source['base_sha256']
    for family in rock.FAMILIES:
        for p in adapters.params(family):p.data=p.data.float()
    adapters.load(source['adapters']);planner=l.Planner().to(model.device);planner.load_state_dict(source['planner']);planner.eval()
    conditions=l.Conditions().to(model.device);conditions.load_state_dict(source['conditions']);conditions.eval();actors={r:g.oe.Musician().eval() for r in rock.ROLES}
    selected=mix.select_windows(windows,'train')+mix.select_windows(windows,'validation');records={};bank=output/'history_bank';bank.mkdir(exist_ok=True)
    meta=dict(source_sha256=source_sha,data_sha256=manifest['examples_sha256'],code_sha256=sha(ROOT/'rock_history_mix.py'),goal=goal)
    write(output/'experiment.json',dict(schema='rock-history-mix-v1',source_step=128,source_sha256=source_sha,arms=['control','mixed'],updates_per_arm=64,total_trainer_budget_seconds=7200,
        common_changes='Both arms add startup supervision and use the training-mean inference goal. Only the treatment replaces histories.',
        mixing='One real bar anchors one frozen-policy generated bar; at most half of phrases requested, rising over 16 updates. Targets stay original; unsupported examples use real history.',
        limitations=['Generated histories do not have oracle expert continuations; this is biased context augmentation, not DAgger.','Frozen rollout bank does not track the changing treatment policy.','Does not prove autonomous quality, cooperation, or happy mood.'],
        references=['https://arxiv.org/abs/1511.05101','https://proceedings.mlr.press/v15/ross11a.html'],evaluation_seeds=[9976501,9976502,9976503],new_rl=False))
    for index,w in enumerate(selected):
        if time.monotonic()>=deadline:raise TimeoutError('Time cap reached during history construction')
        path=bank/(w['id']+'.json')
        if path.exists():record=json.loads(path.read_text());assert record['meta']==meta
        else:
            record=mix.rollin(model,adapters,planner,conditions,actors,w,goal,9976410+index*100003);record['meta']=meta;write(path,record);on_save()
        records[w['id']]=record
        print(json.dumps(dict(event='history_cached',index=index+1,total=len(selected),split=w['split'],group=w['group'],notes=len(record['generated']))),flush=True)
    write(output/'history_bank_manifest.json',dict(meta=meta,windows={w['id']:dict(split=w['split'],group=w['group']) for w in selected}))
    results={}
    for name in ['control','mixed']:
        torch.manual_seed(9976400);random.seed(9976400);np.random.seed(9976400)
        arm=Arm(name,output,model,adapters,planner,conditions,source,windows,records,goal,deadline,on_save)
        results[name]=arm.train();del arm
    assert sha(source_path)==source_sha
    n=min(len(results['control']['updates']),len(results['mixed']['updates']))
    assert all(results['control']['updates'][i]['sample_plan_sha256']==results['mixed']['updates'][i]['sample_plan_sha256'] for i in range(n))
    report=dict(complete=all(s['complete'] for s in results.values()),source_unchanged=True,frozen_backbone_unchanged=adapters.base_hash()==source['base_sha256'],sample_plans_matched=True,
        elapsed_seconds=time.monotonic()-started,arms={k:dict(step=s['step'],selected_step=s['selected_step'],complete=s['complete']) for k,s in results.items()})
    if report['complete'] and time.monotonic()<deadline:
        print(json.dumps(dict(event='matched_evaluation_start')),flush=True)
        report['evaluation']=evaluation(output,source,model,adapters,planner,conditions,actors,goal,deadline,on_save)
        assert adapters.base_hash()==source['base_sha256']
    report['elapsed_seconds']=time.monotonic()-started
    write(output/'result.json',report);on_save();return report
