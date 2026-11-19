"""Bounded anchored context augmentation, never an expert oracle or RL reward."""
import copy,hashlib,json
from pathlib import Path
import torch
import rock_prompt_diagnostic as diagnostic
g=diagnostic.g;rock=g.rock;l=g.l

def eligible_window(w):
    return (not any(1920<=n[1]<3840 and n[1]+n[2]>3840 for n in w['notes'])
        and any(n[1]<1920 and n[3]!=rock.ROLES['drums']['id'] for n in w['notes']))

def select_windows(windows,split):
    by_group={}
    for w in windows:
        if w['split']==split and eligible_window(w):by_group.setdefault(w['group'],[]).append(w)
    selected=[]
    for group in sorted(by_group):
        pool=sorted(by_group[group],key=lambda w:(-len(w['available_roles']),hashlib.sha256(w['id'].encode()).hexdigest()))
        selected.append(pool[0])
        covered=set(pool[0]['available_roles'])
        for role in rock.ROLES:
            if role in covered:continue
            match=next((w for w in pool if role in w['available_roles']),None)
            if match is not None:selected.append(match);covered.update(match['available_roles'])
    return selected

@torch.no_grad()
def rollin(model,adapters,planner,conditions,actors,w,goal,seed):
    """One real bar anchors key/context; sample only the following bar."""
    if not eligible_window(w):raise ValueError('Unsupported source sustain across replaced-history boundary')
    anchor=[copy.deepcopy(n) for n in w['notes'] if n[1]<1920]
    ledger=copy.deepcopy(anchor);made=[];previous=torch.zeros(l.SIZE,device=model.device);hidden=None;plans=[]
    goal=torch.tensor(goal,device=model.device)
    # Exact generation/training two-bar schedule: no hidden access to source bar 2.
    for bar in range(2):
        x=torch.cat([previous,torch.zeros_like(previous),goal,previous.new_tensor([bar/5])])
        pred,hidden=planner(x[None,None],hidden);previous=pred[0,0];plans.append(previous)
    brief=g.nt.Brief(6,w['bpm'],0,False);g.BARS=6
    for beat in range(4,8):
        past=l.visible(ledger,beat);actions={r:[] for r in rock.ROLES}
        for role in w['available_roles']:
            s=rock.ROLES[role];adapters.activate(s['family'])
            delta=conditions(plans[beat//4],torch.zeros(1,8,device=model.device),role,goal)
            sampler=rock.Sampler(g.oe.ConditionedModel(model,delta),role,seed+beat*1009+s['id']*100003,brief,beat,past)
            notes,_=sampler.propose(actors[role],torch.zeros(32))
            for n in notes:n[3]=s['id']
            actions[role]=notes;made.extend(copy.deepcopy(notes))
        ledger=g.commit(ledger,actions,beat)
    assert sorted(ledger)==sorted(anchor+made) and all(1920<=n[1]<3840 for n in made)
    return dict(source_window=w['id'],group=w['group'],split=w['split'],seed=seed,notes=ledger,anchor=anchor,generated=made,
        semantics='Frozen step-128 one-bar predictions anchored by one real bar. No source second-bar or future target notes are given to the sampler.')

def mixed_ledger(w,record):
    if w['id']!=record['source_window'] or w['group']!=record['group'] or w['split']!=record['split']:raise ValueError('Roll-in belongs to a different source/split')
    if any(n[1]>=3840 for n in record['notes']):raise ValueError('Future onset in roll-in')
    return record['notes']+[copy.deepcopy(n) for n in w['notes'] if n[1]>=3840]

def supported_history(history):
    for role,s in rock.ROLES.items():
        events=[]
        for n in history:
            if n[3]==s['id']:events.extend([(n[1],1,n[4]),(n[1]+n[2],0,n[4])])
        active=set()
        for tick,on,pitch in sorted(events):
            if on:
                if pitch in active or len(active)>=s['polyphony']:return False
                active.add(pitch)
            else:active.discard(pitch)
    return True

def plan_sequence(planner,w,goal,record=None):
    if record is None:return l.plan_sequence(planner,w['notes'],6,w['available_roles'],goal)
    ledger=mixed_ledger(w,record);device=next(planner.parameters()).device
    previous=torch.zeros(l.SIZE,device=device);hidden=None;plans=[]
    for start in range(0,24,8):
        history=l.visible(ledger,start)
        if not supported_history(history):history=l.visible(w['notes'],start)
        summary=l.description(history,start//4-1,6,w['available_roles']).to(device) if start else torch.zeros_like(previous)
        for bar in range(start//4,start//4+2):
            x=torch.cat([previous,summary,goal.to(device),previous.new_tensor([bar/5])])
            pred,hidden=planner(x[None,None],hidden);previous=pred[0,0];plans.append(previous)
    return torch.stack(plans)

def examples(model,w,role,record=None):
    cold=[];continuation=[];stats=dict(requested=0,used=0,changed=0,fallback=0,forced_real=0,reasons={})
    ledger=mixed_ledger(w,record) if record is not None else None
    for beat in range(24):
        real=rock.teacher(model,w,role,beat,allow_warmup=True)
        if real is None:continue
        if beat<8:cold.append(real);continue
        ex=real;used=False
        if ledger is not None:
            stats['requested']+=1
            try:
                history=l.visible(ledger,beat)
                if not supported_history(history):raise ValueError('Hybrid history violates playability')
                candidate=rock.teacher(model,w,role,beat,history_override=history)
                if candidate is None:raise ValueError('Generated history removes a free decision')
                ex=candidate;used=True;stats['used']+=1
                stats['changed']+=not torch.equal(candidate['x'],real['x'])
            except ValueError as exc:
                reason=str(exc)
                if not (reason.startswith('Unsupported ') or reason.startswith('Target under ') or reason in ['Generated history removes a free decision','Hybrid history violates playability']):raise
                stats['fallback']+=1;stats['reasons'][reason]=stats['reasons'].get(reason,0)+1
        else:stats['forced_real']+=1
        continuation.append({**ex,'history_mixed':used})
    return cold,continuation,stats
