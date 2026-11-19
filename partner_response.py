"""Causal fixed-partner curriculum and full-policy supervised response records.

Source partners are labelled dataset notes. Learners never see current/future
partner notes before commit. Final joint performances contain only policy notes.
"""
import copy
import gzip
import hashlib
import json
import math
import random
from collections import Counter

import numpy as np
import torch
import note_trio as nt
import ensemble_data as ed
import supervised_ensemble as se
from sequence_critic import critic_state


def identity(example):
    return example['song'],example['window_start']


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def response_windows(train_limit=64,validation_limit=12):
    audit=json.loads((ed.DATA/'audit.json').read_text())
    if audit['format_version']<2:raise ValueError('Prepare ensemble data v2 first')
    output={s:{r:[] for r in nt.ROLES} for s in ['train','validation']}
    rng=random.Random(20261018)
    for split in output:
        songs=[r for r in audit['kept_songs'] if r['split']==split]
        rng.shuffle(songs);candidates={r:{} for r in nt.ROLES}
        for song in songs:
            with gzip.open(ed.DATA/'corpus'/f'{song["song"]}.json.gz','rt',encoding='utf-8') as f:record=json.load(f)
            ws=list(se.windows(record));rng.shuffle(ws)
            for role in nt.ROLES:
                usable=[];channel=nt.ROLES[role]['channel']
                for w in ws:
                    if role not in w['available_roles']:continue
                    # Both a causal partner prefix and a sounding target suffix.
                    if not any(n[3]!=channel and n[1]<1920 for n in w['notes']):continue
                    if not any(n[3]==channel and n[1]>=1920 for n in w['notes']):continue
                    usable.append(w)
                if usable:candidates[role][song['song']]=usable[:8]
        limit=train_limit if split=='train' else validation_limit
        for role,by_song in candidates.items():
            while len(output[split][role])<limit and any(by_song.values()):
                for ws in by_song.values():
                    if ws:output[split][role].append(ws.pop())
                    if len(output[split][role])==limit:break
            if len(output[split][role])<(8 if split=='train' else 3):
                raise ValueError(f'Insufficient fixed-partner windows for {split}/{role}')
    train={w['song'] for ws in output['train'].values() for w in ws}
    validation={w['song'] for ws in output['validation'].values() for w in ws}
    assert not train&validation
    hashes={s:{fingerprint(w['notes']) for ws in roles.values() for w in ws} for s,roles in output.items()}
    # Broader window coverage can reveal repeated passages across song IDs.
    for role in nt.ROLES:
        output['train'][role]=[w for w in output['train'][role] if fingerprint(w['notes']) not in hashes['validation']]
        if len(output['train'][role])<8:raise ValueError('Too few training windows after exact-passage exclusion')
    assert not ({fingerprint(w['notes']) for ws in output['train'].values() for w in ws}&hashes['validation'])
    return output


def donor_for(example,role,pool):
    """Different composition, same retained partner channels, same beat timeline."""
    partners=set(example['available_roles'])-{role}
    eligible=[w for w in pool if w['song']!=example['song'] and partners<=set(w['available_roles'])]
    if not eligible:raise ValueError(f'No composition-disjoint partner donor for {role}')
    digest=int(fingerprint([*identity(example),role])[:12],16)
    return eligible[digest%len(eligible)]


def observation_history(ledger,beat,role,mode='correct',donor=None,partner_roles=None):
    history=nt.visible(ledger,beat);channel=nt.ROLES[role]['channel']
    if mode=='correct':return history
    own=[n for n in history if n[3]==channel]
    if mode=='missing':return own
    if mode!='shuffled' or donor is None:raise ValueError('Shuffled history requires a donor')
    channels={nt.ROLES[r]['channel'] for r in (partner_roles or nt.ROLES) if r!=role}
    return sorted(own+[n for n in nt.visible(donor['notes'],beat) if n[3] in channels],key=lambda n:(n[1],n[3],n[4]))


@torch.no_grad()
def teacher_records(model,adapters,role,example,mode='correct',donor=None):
    history=observation_history(example['notes'],example['beat'],role,mode,donor,example['available_roles'])
    ex=se.teacher_example(model,role,example,history=history)
    if ex is None:return None
    adapters.activate(role)
    hidden=model(ex['x'].to(model.device).unsqueeze(0))[0,ex['positions'].to(model.device)]
    y=ex['labels'].to(model.device)
    logits=model.forward_token(hidden,y[:,:-1]).float()
    state=nt.state_features(nt.Brief(4),example['beat'],history)
    observations=[];references=[];actions=[];kinds=[]
    for row in range(len(y)):
        for j,kind in enumerate(se.KINDS):
            ids=ex['support'][row,j].nonzero().flatten().tolist()
            if not ids:continue
            if kind=='event':mapping={model.tokenizer.event_ids['note']:0,model.tokenizer.eos_id:1}
            else:mapping={token:value for value,token in enumerate(model.tokenizer.parameter_ids[kind])}
            values=[mapping[token] for token in ids]
            reference=torch.zeros(128)
            reference[values]=.98*torch.softmax(logits[row,j,ids]/.9,-1).cpu()+.02/len(ids)
            observation=torch.zeros(1072);observation[:1024]=hidden[row].float().cpu()
            observation[1024:1056]=state;observation[1056+j]=1
            observation[1064:1064+j]=y[row,:j].float().cpu()/model.tokenizer.vocab_size
            observations.append(observation);references.append(reference);actions.append(mapping[int(y[row,j])]);kinds.append(j)
    if not actions:return None
    return dict(observations=torch.stack(observations),references=torch.stack(references),actions=torch.tensor(actions),
                kinds=torch.tensor(kinds),song=example['song'],window_start=example['window_start'],beat=example['beat'],mode=mode)


def policy_nll(actor,records):
    return -actor.distribution(records['observations'],records['references']).log_prob(records['actions'])


def concatenate(records):
    return {k:torch.cat([r[k] for r in records]) for k in ['observations','references','actions','kinds']}


def build_teacher_cache(model,adapters,data,train_beats=256,val_beats=48):
    cache={s:{r:[] for r in nt.ROLES} for s in data}
    for split,roles in data.items():
        for role,windows in roles.items():
            rng=random.Random(20261018+list(nt.ROLES).index(role))
            examples=[]
            for w in windows:
                beats=list(range(4,16));rng.shuffle(beats)
                for beat in beats[:6]:examples.append({**w,'beat':beat})
            rng.shuffle(examples);examples=examples[:train_beats if split=='train' else val_beats]
            for i,ex in enumerate(examples):
                try:donor=donor_for(ex,role,windows)
                except ValueError:
                    print(f'Skipping unmatched donor {split}/{role}/{ex["song"]}',flush=True)
                    continue
                pair={mode:teacher_records(model,adapters,role,ex,mode,donor) for mode in (['correct','shuffled'] if split=='train' else ['correct','missing','shuffled'])}
                if any(r is None for r in pair.values()):continue
                for other in pair.values():
                    assert torch.equal(pair['correct']['actions'],other['actions'])
                    assert torch.equal(pair['correct']['references']>0,other['references']>0)
                cache[split][role].append(pair)
                if (i+1)%64==0:print(f'Response cache {split}/{role}: {i+1}/{len(examples)} beats',flush=True)
            if len(cache[split][role])<4:raise ValueError(f'Too few teacher decisions for {split}/{role}')
            print(f'Response cache {split}/{role}: {len(cache[split][role])} usable beats',flush=True)
    return cache


@torch.no_grad()
def evaluate_cache(agents,cache):
    results={}
    for role,pairs in cache.items():
        results[role]={}
        for mode in ['correct','missing','shuffled']:
            records=concatenate([pair[mode] for pair in pairs]);loss=policy_nll(agents[role],records)
            results[role][mode]=dict(nll=float(loss.mean()),decisions=len(loss))
            results[role][mode]['by_decision']={se.KINDS[int(kind)]:dict(nll=float(loss[records['kinds']==kind].mean()),
                decisions=int((records['kinds']==kind).sum())) for kind in records['kinds'].unique()}
        results[role]['correct_better_than_missing']=results[role]['correct']['nll']<results[role]['missing']['nll']
        results[role]['correct_better_than_shuffled']=results[role]['correct']['nll']<results[role]['shuffled']['nll']
    return results


def ensemble_score(notes,end,roles=None):
    """Bounded diagnostics for active roles; no missing-role participation penalty."""
    roles=list(nt.ROLES) if roles is None else list(roles)
    channels={nt.ROLES[r]['channel'] for r in roles};left=max(0,end-1920)
    notes=[n for n in notes if n[1]<end and n[3] in channels]
    active=[n for n in notes if n[1]+n[2]>left]
    if not active:return -1.,dict(silence=1.)
    harmony=weight=0.
    for i,a in enumerate(active):
        for b in active[i+1:]:
            if a[3]==b[3]:continue
            overlap=max(0,min(end,a[1]+a[2],b[1]+b[2])-max(left,a[1],b[1]))
            interval=abs(a[4]-b[4])%12
            harmony+=overlap*(1. if interval in [0,3,4,5,7,8,9] else -.5)
            weight+=overlap
    harmony=harmony/max(weight,1)
    coverage=[];density=[];motion=[];repetition=[]
    for role in roles:
        own=[n for n in notes if n[3]==nt.ROLES[role]['channel']]
        sounding=[any(n[1]<=tick<n[1]+n[2] for n in own) for tick in range(left,end,60)]
        coverage.append(min(1,sum(sounding)/max(1,len(sounding))/.6))
        attacks=sum(left<=n[1]<end for n in own)/4
        upper={'organ':2.5,'strings':1.,'cello':1.25}[role]
        density.append(1-min(1,max(0,attacks-upper)/upper))
        recent=sorted([n for n in own if n[1]>=end-7680],key=lambda n:(n[1],n[4]))
        line=[max(n[4] for n in recent if n[1]==t) for t in sorted({n[1] for n in recent})]
        if len(line)>1:motion.append(float(np.mean([math.exp(-max(0,abs(a-b)-7)/7) for a,b in zip(line,line[1:])])) )
        if len(line)>=8:repetition.append(max(0,(max(Counter(line).values())/len(line)-.8)/.2))
    parts=dict(cross_role_harmony=float(harmony),coverage=float(np.mean(coverage)),density=float(np.mean(density)),
               voice_leading=float(np.mean(motion)) if motion else 0.,degeneracy=float(np.mean(repetition)) if repetition else 0.)
    score=.45*parts['cross_role_harmony']+.25*parts['coverage']+.1*parts['density']+.2*parts['voice_leading']-.3*parts['degeneracy']
    return float(np.clip(score,-1,1)),parts


def response_rollout(model,adapters,actor,critic,role,example,seed,mode='correct',donor=None):
    brief=nt.Brief(4);channel=nt.ROLES[role]['channel']
    ledger=nt.visible(example['notes'],4);steps=[];decisions=[];trace=[]
    prefix=copy.deepcopy(ledger)
    for beat in range(4,16):
        true_history=nt.visible(ledger,beat)
        history=observation_history(ledger,beat,role,mode,donor,example['available_roles'])
        state=nt.state_features(brief,beat,history);cs=critic_state(brief,beat,true_history)
        with torch.no_grad():value=float(critic(cs))
        adapters.activate(role)
        sampler=se.EnsembleSampler(model,role,seed+beat*1009+list(nt.ROLES).index(role)*100003,brief,beat,history)
        notes,records=sampler.propose(actor,state)
        actions={r:[] for r in nt.ROLES};actions[role]=notes
        fixed=[copy.deepcopy(n) for n in example['notes'] if n[3]!=channel and n[1]//480==beat]
        for r,s in nt.ROLES.items():
            if r!=role:actions[r]=[n for n in fixed if n[3]==s['channel']]
        ledger=nt.commit(ledger,actions,beat,brief)
        decisions.extend(dict(role=role,beat=beat,note=copy.deepcopy(n),origin='learner_policy') for n in notes)
        trace.append(dict(role=role,beat=beat,mode=mode,history_hash=fingerprint(history),
                          observed_partner_notes=sum(n[3]!=channel for n in history),fixed_partner_notes=len(fixed)))
        score,parts=ensemble_score(ledger,(beat+1)*480,example['available_roles']) if beat%4==3 else (0.,{})
        steps.append(dict(state=state,critic_state=cs,value=value,reward=score,components=parts,records={role:records}))
    learned=[n for n in ledger if n[3]==channel and n[1]>=1920]
    assert sorted(learned)==sorted(d['note'] for d in decisions)
    return dict(notes=ledger,steps=steps,trace=trace,decisions=decisions,brief=nt.asdict(brief),seed=seed,
                dataset_prefix=prefix,dataset_fixed_partners=[n for n in ledger if n[3]!=channel and n[1]>=1920],
                learner=role,teacher_partner_roles=[r for r in example['available_roles'] if r!=role],song=example['song'],window_start=example['window_start'])


def joint_rollout(model,adapters,agents,critic,brief,seed,mode='correct',order=None):
    order=list(nt.ROLES) if order is None else list(order)
    if sorted(order)!=sorted(nt.ROLES):raise ValueError('Every role must propose once')
    if mode not in ['correct','missing']:raise ValueError('Joint inference supports correct/missing history')
    ledger=[];steps=[];trace=[];decisions=[]
    for beat in range(brief.bars*4):
        true_history=nt.visible(ledger,beat);cs=critic_state(brief,beat,true_history)
        with torch.no_grad():value=float(critic(cs))
        actions={};records={}
        for role in order:
            history=observation_history(ledger,beat,role,mode)
            state=nt.state_features(brief,beat,history)
            adapters.activate(role)
            sampler=se.EnsembleSampler(model,role,seed+beat*1009+list(nt.ROLES).index(role)*100003,brief,beat,history)
            actions[role],records[role]=sampler.propose(agents[role],state)
            decisions.extend(dict(role=role,beat=beat,note=copy.deepcopy(n),origin='joint_policy') for n in actions[role])
            trace.append(dict(role=role,beat=beat,mode=mode,history_hash=fingerprint(history),
                              observed_partner_notes=sum(n[3]!=nt.ROLES[role]['channel'] for n in history)))
        ledger=nt.commit(ledger,actions,beat,brief)
        score,parts=ensemble_score(ledger,(beat+1)*480) if beat%4==3 else (0.,{})
        # All actors receive the same committed history during joint training.
        state=nt.state_features(brief,beat,true_history)
        steps.append(dict(state=state,critic_state=cs,value=value,reward=score,components=parts,records=records))
    assert sorted(ledger)==sorted(d['note'] for d in decisions)
    return dict(notes=ledger,steps=steps,trace=trace,decisions=decisions,brief=nt.asdict(brief),seed=seed)
