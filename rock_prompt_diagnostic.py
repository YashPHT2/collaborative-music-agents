"""Six-bar, same-seed context diagnostic; reference notes explicitly attributed."""
import copy,gzip,hashlib,json
from pathlib import Path
import numpy as np,torch
import generate_rock_comparison as g

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs/rock_prompt_diagnostic_v1'
SEED=9976201;CHECKPOINTS={'step32':'selected.pt','step128':'progress.pt'}

def prepare():
    OUT.mkdir(exist_ok=True)
    with gzip.open(ROOT/'outputs/rock_training_v1/rock_examples.json.gz','rt') as f:windows=json.load(f)
    eligible=[w for w in windows if w['split']=='validation' and not any(n[1]<3840<n[1]+n[2] for n in w['notes'])
        and all(any(n[3]==r and n[1]<3840 for n in w['notes']) for r in range(5))]
    eligible.sort(key=lambda w:hashlib.sha256(w['id'].encode()).hexdigest());w=eligible[0]
    goal=np.mean([g.l.style([n for n in row['notes'] if n[3]!=3],row['available_roles']).numpy() for row in windows if row['split']=='train'],axis=0).tolist()
    prefix=[copy.deepcopy(n) for n in w['notes'] if n[1]<3840]
    protocol=dict(schema='rock-prompt-diagnostic-v1',seed=SEED,bars=6,prefix_bars=2,target_bars=4,bpm=w['bpm'],
        prompt_source_window=w['id'],prompt_group=w['group'],prompt_split='validation',prefix=prefix,goal=goal,
        selection='First SHA-ordered eligible validation window; all five roles present in prefix, no note sustained across prompt boundary. No quality scoring.',
        new_training=False,checkpoint_sha256={k:g.sha(ROOT/'outputs/rock_training_v1/checkpoints'/('selected.pt' if k=='step32' else 'final128.pt')) for k in CHECKPOINTS},
        limitation='One validation prefix and one seed per checkpoint. Exploratory context diagnostic; not a held-out quality certification.')
    path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    g.write_json(path,protocol)
    # Kept locally for later diagnostics; never sent as part of inference input.
    g.write_json(OUT/'reference_window.json',w)
    print('Prepared prefix:',len(prefix),'notes;',round(8*60/w['bpm'],3),'seconds; BPM',w['bpm'],flush=True)
    return protocol

def check_prefix(prefix):
    if not prefix or any(n[1]<0 or n[1]>=3840 or n[1]+n[2]>3840 for n in prefix):raise ValueError('Prompt contains future or crossing notes')
    return copy.deepcopy(prefix)

@torch.no_grad()
def compose(model,adapters,planner,conditions,actors,p,condition):
    if condition not in ['unprompted','prompted']:raise ValueError('Unknown condition')
    prefix=check_prefix(p['prefix']) if condition=='prompted' else []
    ledger=copy.deepcopy(prefix);decisions=[];plans=[];trace=[]
    goal=torch.tensor(p['goal'],device=model.device);previous=torch.zeros(g.l.SIZE,device=model.device);hidden=None
    brief=g.nt.Brief(6,p['bpm'],0,False);g.BARS=6
    for start in range(0,24,8):
        summary=g.l.description(g.l.visible(ledger,start),start//4-1,6,list(g.rock.ROLES)).to(model.device) if start else torch.zeros_like(previous)
        for bar in range(start//4,start//4+2):
            x=torch.cat([previous,summary,goal,previous.new_tensor([bar/5])]);pred,hidden=planner(x[None,None],hidden);previous=pred[0,0];plans.append(previous)
        if prefix and start==0:continue
        for beat in range(start,start+8):
            past=g.l.visible(ledger,beat);assert all(n[1]<beat*480 for n in past)
            actions={}
            for role,s in g.rock.ROLES.items():
                adapters.activate(s['family'])
                delta=conditions(plans[beat//4],torch.zeros(1,8,device=model.device),role,goal)
                sampler=g.rock.Sampler(g.oe.ConditionedModel(model,delta),role,p['seed']+beat*1009+s['id']*100003,brief,beat,past)
                made,_=sampler.propose(actors[role],torch.zeros(32))
                for n in made:n[3]=s['id']
                actions[role]=made;decisions.extend(dict(role=role,beat=beat,note=n,origin='agent') for n in made)
            ledger=g.commit(ledger,actions,beat)
        trace.append(dict(start=start,total_notes=len(ledger)))
        print(condition,start//4+2,'/6 bars',len(ledger),'notes',flush=True)
    assert sorted(ledger)==sorted(prefix+[d['note'] for d in decisions])
    if prefix:assert sorted(n for n in ledger if n[1]<3840)==sorted(prefix)
    return dict(notes=ledger,decisions=decisions,reference_prefix=prefix,trace=trace,condition=condition,seed=p['seed'],bars=6,bpm=p['bpm'],
        reference_ends_at_seconds=8*60/p['bpm'],generated_continuation_notes=sum(d['beat']>=8 for d in decisions),
        attribution='Reference prefix is supplied data only in prompted condition; all new onsets after beat 8 are agent decisions.')

if __name__=='__main__':g.rock.configure();prepare()
