"""Verified prefix/target adapter to the production teacher sampler.

No eight-note eligibility filter. Prefix events are context only, partner future
notes never enter teacher history, and absent roles are masked rather than rests.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

import audit_supported_phrase_gate as gate
import ensemble_repair as er
import offline_ensemble as oe
import supervised_ensemble as se
import note_trio as nt

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'outputs/supported_corpus_expansion_v1'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def load_data(directory=DATA,minimum=12):
    directory=Path(directory)
    manifest=read(directory/'summary.json')
    path=directory/'eligible_examples.json'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['eligible_examples_sha256']:
        raise ValueError('Expanded examples differ from audited manifest')
    rows=read(path);examples=[]
    for row in rows:
        if not row['eligible'] or row['notes'] is None or row['unsupported_source_notes']:
            raise ValueError('Ineligible/unsupported example reached training loader')
        notes=row['notes'];loss_start=row['prefix_ticks']//480;bars=(row['prefix_ticks']+row['target_ticks'])//1920
        available=[r for r in er.ROLES if row['roles'][r]['loss_mask']]
        example={**row,'bars':bars,'loss_start_beat':loss_start,'loss_end_beat':loss_start+16,
                 'available_roles':available,'source_roles':list(row['assignment']),
                 'fingerprint':row['id'],'window_start':row['start_tick'],
                 'role_weights':{r:1. if row['roles'][r]['provenance']=='exact' else .35 for r in available}}
        assert len(row['loss_beat_mask'])==bars*4
        assert all(not row['loss_beat_mask'][b] for b in range(loss_start))
        examples.append(example)
    groups={s:{r:defaultdict(list) for r in er.ROLES} for s in ('train','validation','test')}
    for w in examples:
        for role in w['available_roles']:groups[w['split']][role][w['group']].append(w)
    counts={r:len(groups['validation'][r]) for r in er.ROLES}
    if any(v<minimum for v in counts.values()):raise ValueError(f'Need {minimum} independent validation compositions per role, got {counts}')
    partitions={s:{w['group'] for w in examples if w['split']==s} for s in groups}
    if any(partitions[a]&partitions[b] for a,b in [('train','validation'),('train','test'),('validation','test')]):
        raise ValueError('Composition group crosses partitions')
    return examples,groups,manifest


def teacher(model,w,role,beat):
    if not w['loss_start_beat']<=beat<w['loss_end_beat']:return None
    if role not in w['available_roles']:return None
    history=nt.visible(w['notes'],beat)
    if any(n[1]>=beat*480 for n in history):raise ValueError('Future note leaked into teacher history')
    with oe.limits():
        ex=se.teacher_example(model,role,{**w,'beat':beat,'sampler_type':oe.Sampler},history=history)
    if ex is not None:
        ex['kind']=er.beat_kind(w,role,beat)
        ex['beat']=beat
        ex['history_note_count']=len(history)
        ex['provenance']=w['roles'][role]['provenance']
    return ex


def validation_set(groups,role,maximum=12):
    """Fixed groups, source-provenance stratification where both strata exist."""
    rng=random.Random(989010+er.ROLES.index(role));keys=sorted(groups[role]);rng.shuffle(keys)
    selected=[]
    for provenance in ('exact','transfer'):
        matches=[key for key in keys if any(w['roles'][role]['provenance']==provenance for w in groups[role][key])]
        for key in matches[:maximum//2]:
            if key not in selected:selected.append(key)
    for key in keys:
        if len(selected)>=maximum:break
        if key not in selected:selected.append(key)
    result=[]
    for key in selected:
        windows=sorted(groups[role][key],key=lambda w:w['id'])
        result.append(windows[rng.randrange(len(windows))])
    return result


def supported_categories(ex):
    mask=ex['support'].any(-1)
    keys=[k for k,column in [('pitch',5),('onset',2),('duration',7),('velocity',6)] if mask[:,column].any()]
    if mask[0,0]:keys.append('first_event')
    if mask[1:,0].any():keys.append('additional_event')
    return keys


def balance_weights(examples,role_weight=1.,normalizer=1.):
    """Equal supported categories and onset/non-onset strata within a phrase."""
    from collections import Counter
    counts={k:Counter() for k in er.CATEGORIES}
    for ex in examples:
        for k in supported_categories(ex):counts[k][ex['kind']=='onset']+=1
    categories=sum(bool(v) for v in counts.values())
    return [{k:role_weight/(normalizer*categories*len(counts[k])*counts[k][ex['kind']=='onset'])
             for k in supported_categories(ex)} for ex in examples]


def selection_eligible(scores,baseline):
    if not er.eligible({r:scores[r]['all'] for r in er.ROLES},{r:baseline[r]['all'] for r in er.ROLES}):return False
    for role in er.ROLES:
        for provenance in ('exact','transfer'):
            current=scores[role].get(provenance);old=baseline[role].get(provenance)
            if not current or not old:continue
            for metric in ('event','attributes'):
                if old[metric] is not None and (current[metric] is None or current[metric]>old[metric]*1.02):return False
    return True
