"""Causal ensemble imitation through three LoRA adapters, then sequence-critic PPO.

Teacher labels use the exact legal support of NoteSampler. Only committed past
partner notes and the role's own earlier notes in the current beat are exposed.
"""
import argparse
import copy
import gzip
import hashlib
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
import note_trio as nt
from peft import LoraConfig,LoraModel
import ensemble_data as ed
from sequence_critic import SequenceCritic,critic_state,INPUT as CRITIC_INPUT

OUT=nt.ROOT/'outputs'/'supervised_trio_v1'
VERSION=1
KINDS=['event','time1','time2','track','channel','pitch','velocity','duration']


class InstrumentAdapters:
    def __init__(self,model):
        self.model=model
        configs={r:LoraConfig(r=8,lora_alpha=16,lora_dropout=0.,target_modules=['q_proj','v_proj']) for r in nt.ROLES}
        self.wrapper=LoraModel(model,configs,adapter_name='organ')
        for role in ['strings','cello']:
            self.wrapper.inject_adapter(model,role)
        self.model.net.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        self.activate('organ')

    def params(self,role):
        return [p for n,p in self.model.named_parameters() if 'lora_' in n and f'.{role}.' in n]

    def activate(self,role,training=False):
        self.wrapper.set_adapter(role);self.model.requires_grad_(False)
        self.model.train(training)
        if training:
            for p in self.params(role):p.requires_grad_(True)

    def state(self):
        return {k:v.detach().cpu().clone() for k,v in self.model.state_dict().items() if 'lora_' in k}

    def load(self,state):
        if set(state)!=set(self.state()):raise ValueError('Wrong LoRA instrument state')
        self.model.load_state_dict(state,strict=False)

    def base_hash(self):
        h=hashlib.sha256()
        for n,p in self.model.state_dict().items():
            if 'lora_' not in n:h.update(n.encode());h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()


def load_bundle(path):
    """Load a local supervised-plus-PPO checkpoint for new compositions."""
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved.get('format_version')!=VERSION or saved.get('roles')!=nt.ROLES:
        raise ValueError('Incompatible ensemble checkpoint')
    if saved.get('critic_type')!='sequence-gru-v1' or saved.get('critic_input')!=CRITIC_INPUT:
        raise ValueError('Wrong sequence critic format')
    model=nt.load_model();adapters=InstrumentAdapters(model);adapters.load(saved['adapters'])
    agents={r:nt.Musician() for r in nt.ROLES}
    for r,a in agents.items():a.load_state_dict(saved['actors'][r]);a.eval()
    critic=SequenceCritic();critic.load_state_dict(saved['critic']);critic.eval()
    return model,adapters,agents,critic


def playable_projection(notes):
    """Prune source voices to agent limits; never invent pitches or onsets.

    Real string tracks may be chords and real cello tracks may be sections.
    Upper strings retain higher chord voices; cello retains the lowest voice
    and shortens a held note when the next playable note starts.
    """
    result=[]
    for role,settings in nt.ROLES.items():
        own=[];counts=Counter()
        source=sorted([copy.deepcopy(n) for n in notes if n[3]==settings['channel']],
                      key=lambda n:(n[1],-n[4] if role=='strings' else n[4]))
        for n in source:
            if counts[n[1]//480]>=settings['cap']:continue
            active=[p for p in own if p[1]<=n[1]<p[1]+p[2]]
            if role=='cello' and active:
                previous=active[-1]
                duration=(n[1]-previous[1])//60*60
                if duration<60:continue
                previous[2]=min(previous[2],duration);active=[]
            if len(active)>=settings['polyphony'] or n[4] in {p[4] for p in active}:continue
            own.append(n);counts[n[1]//480]+=1
        result.extend(own)
    return sorted(result,key=lambda n:(n[1],n[3],n[4]))


def windows(record):
    """Four-bar examples, with support-valid notes and a one-bar causal prefix."""
    notes=record['notes'];end=max((n[1]+n[2] for n in notes),default=0)
    for start in range(0,max(0,end-4*1920+1),4*1920):
        stop=start+4*1920;ns=[]
        for n in notes:
            if not (start<=n[1]<stop):continue
            m=copy.deepcopy(n);m[1]-=start;m[2]=min(m[2],stop-n[1])//60*60
            if m[2]>=60:ns.append(m)
        present=[r for r,s in nt.ROLES.items() if any(n[3]==s['channel'] for n in ns)]
        if len(present)<2:continue
        source_count=len(ns);ns=playable_projection(ns)
        ledger=[];valid=True
        for beat in range(16):
            actions={r:[n for n in ns if n[3]==s['channel'] and n[1]//480==beat] for r,s in nt.ROLES.items()}
            try:ledger=nt.commit(ledger,actions,beat,nt.Brief())
            except ValueError:valid=False;break
        if valid:yield dict(song=record['song'],split=record['split'],window_start=start,notes=ledger,source_roles=record['roles'],available_roles=present,
                            projection=dict(source_notes=source_count,retained_notes=len(ns)))


def make_examples(limit=256,val_limit=32):
    audit=json.loads((ed.DATA/'audit.json').read_text())
    if audit['format_version']<2:raise ValueError('Re-run ensemble_data.py for audited, missing-role-aware data')
    result={'train':[],'validation':[]};stats=Counter()
    ids={split:[r['song'] for r in audit['kept_songs'] if r['split']==split] for split in result}
    rng=random.Random(20261103)
    for split in result:
        rng.shuffle(ids[split]);maximum=limit if split=='train' else val_limit
        # At most four examples from a song, then distribute across the corpus.
        candidates={r:[] for r in nt.ROLES}
        for song in ids[split]:
            with gzip.open(ed.DATA/'corpus'/f'{song}.json.gz','rt',encoding='utf-8') as f:record=json.load(f)
            ws=list(windows(record));rng.shuffle(ws);stats[split+'_valid_windows']+=len(ws)
            for w in ws[:4]:
                for role in w['available_roles']:
                    beats=sorted({n[1]//480 for n in w['notes'] if n[3]==nt.ROLES[role]['channel'] and n[1]//480>=4})
                    if beats:candidates[role].append({**w,'beat':rng.choice(beats)})
        for cs in candidates.values():rng.shuffle(cs)
        seen=set()
        while len(result[split])<maximum and any(candidates.values()):
            for cs in candidates.values():
                if not cs:continue
                w=cs.pop();identity=(w['song'],w['window_start'],w['beat'])
                if identity not in seen:seen.add(identity);result[split].append(w)
                if len(result[split])==maximum:break
        stats[split+'_examples_per_role']={r:sum(r in w['available_roles'] for w in result[split]) for r in nt.ROLES}
        stats[split+'_source_notes']=sum(w['projection']['source_notes'] for w in result[split])
        stats[split+'_retained_notes']=sum(w['projection']['retained_notes'] for w in result[split])
        stats[split+'_sounding_targets_per_role']={r:sum(any(n[3]==nt.ROLES[r]['channel'] and n[1]//480==w['beat'] for n in w['notes']) for w in result[split]) for r in nt.ROLES}
        minimum=12 if split=='train' else 4
        if min(stats[split+'_sounding_targets_per_role'].values(),default=0)<minimum:
            raise ValueError(f'Insufficient real sounding ensemble targets: {stats}. No proxy fallback is enabled.')
    # Entire compositions remain in their official split.
    assert not ({r['song'] for r in result['train']}&{r['song'] for r in result['validation']})
    fingerprints={s:{hashlib.sha256(json.dumps(w['notes']).encode()).hexdigest() for w in ws} for s,ws in result.items()}
    if fingerprints['train']&fingerprints['validation']:raise ValueError('Projected exact window duplicate across adaptation splits')
    return result,dict(stats)


def teacher_example(model,role,example,history=None):
    # Missing source instruments are unknown labels, not examples of silence.
    if role not in example.get('available_roles',nt.ROLES):return None
    beat=example['beat'];brief=nt.Brief(example.get('bars',4),100,4,True)
    # Training key labels are absent rather than made-up: omit the key header in
    # both this corpus-conditioning mode and its generation sampler below.
    history=nt.visible(example['notes'],beat) if history is None else copy.deepcopy(history)
    if any(n[1]>=beat*480 for n in history):raise ValueError('Teacher history contains current/future notes')
    sampler=example.get('sampler_type',EnsembleSampler)(model,role,0,brief,beat,history)
    target=sorted([copy.deepcopy(n) for n in example['notes'] if n[3]==nt.ROLES[role]['channel'] and n[1]//480==beat],key=lambda n:(n[1],n[4]))
    labels=[];supports=[];positions=[]
    if not sampler.onsets():
        if target:raise ValueError('Teacher note during forced rest')
        return None
    for index in range(len(target)+1):
        if index==len(target) and len(target)==sampler.settings['cap']:break
        if not sampler.onsets():
            if index<len(target):raise ValueError('Invalid teacher support')
            break
        if index<len(target):
            n=target[index];values=dict(event=0,time1=beat-sampler.last_tick//480,time2=n[1]%480//30,
                                      track=sampler.settings['track'],channel=n[3],pitch=n[4],velocity=n[5],duration=n[2]//30)
        else:values=dict(event=1)
        row=[sampler.tok.pad_id]*8;mask=torch.zeros(8,sampler.tok.vocab_size,dtype=torch.bool)
        decoded={}
        for j,kind in enumerate(KINDS):
            legal,ids=sampler.legal_parameters(kind,decoded)
            if values[kind] not in legal:raise ValueError(f'Teacher outside support: {role}/{kind}/{values[kind]}')
            row[j]=ids[legal.index(values[kind])]
            if len(ids)>1:mask[j,ids]=True
            decoded[kind]=values[kind]
            if j==0 and values[kind]==1:break
        positions.append(len(sampler.rows)-1);labels.append(row);supports.append(mask)
        if index<len(target):sampler.observed.append(n);sampler.append_note(n,sampler.settings['track'])
    if not labels:return None
    return dict(x=torch.tensor(sampler.rows,dtype=torch.long),positions=torch.tensor(positions),
                labels=torch.tensor(labels),support=torch.stack(supports))


class EnsembleSampler(nt.NoteSampler):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # A neutral key prompt is consistent with a corpus without key labels.
        self.rows=[r for r in self.rows if r[0]!=self.tok.event_ids['key_signature']]


def teacher_loss(model,ex):
    device=model.device;x=ex['x'].to(device).unsqueeze(0)
    hidden=model(x)[0,ex['positions'].to(device)]
    y=ex['labels'].to(device);logits=model.forward_token(hidden,y[:,:-1]).float()
    support=ex['support'].to(device);valid=support.any(-1)
    selected=logits[valid];legal=support[valid];labels=y[valid]
    probabilities=torch.softmax((selected/.9).masked_fill(~legal,-torch.inf),-1)
    probabilities=.98*probabilities+.02*legal/legal.sum(-1,keepdim=True)
    nll=-probabilities.gather(1,labels[:,None]).clamp_min(1e-30).log().squeeze(1)
    return nll.mean(),len(nll)


def evaluate_teacher(model,adapters,role,examples,shared=True):
    adapters.activate(role);total=count=0
    with torch.no_grad():
        for example in examples:
            if not shared:
                example={**example,'notes':[n for n in example['notes'] if n[3]==nt.ROLES[role]['channel']]}
            ex=teacher_example(model,role,example)
            if ex is not None:
                loss,n=teacher_loss(model,ex);total+=float(loss)*n;count+=n
    if count==0:raise ValueError(f'No validation decisions for {role}')
    return dict(nll=total/count,decisions=count)


def supervised_train(model,adapters,examples,epochs=2,resume=False):
    initial=adapters.state();logs={};optimizers={r:torch.optim.AdamW(adapters.params(r),lr=1e-4,weight_decay=.01) for r in nt.ROLES}
    selected=copy.deepcopy(initial)
    if resume and (OUT/'sft_progress.pt').exists():
        saved=torch.load(OUT/'sft_progress.pt',map_location='cpu',weights_only=False)
        if saved['epochs']!=epochs or saved['examples']!=len(examples['train']):raise ValueError('Supervised resume settings differ')
        initial=saved['initial'];selected=saved['selected'];logs=saved['logs'];adapters.load(saved['adapters'])
        for r,o in optimizers.items():o.load_state_dict(saved['optimizers'][r])
    for role in nt.ROLES:
        if role not in logs:
            start=evaluate_teacher(model,adapters,role,examples['validation'])
            logs[role]=dict(initial_validation=start,initial_without_partners=evaluate_teacher(model,adapters,role,examples['validation'],shared=False),epochs=[],selected_epoch=0)
        best=min([logs[role]['initial_validation']['nll']]+[l['validation']['nll'] for l in logs[role]['epochs']])
        print(f'{role}: best validation NLL {best:.4f}; {len(logs[role]["epochs"])} epochs completed',flush=True)
        optimizer=optimizers[role]
        for epoch in range(len(logs[role]['epochs']),epochs):
            order=list(examples['train']);random.Random(20261103+epoch).shuffle(order)
            adapters.activate(role,training=True);optimizer.zero_grad();running=[];accum=0
            for i,example in enumerate(order):
                ex=teacher_example(model,role,example)
                if ex is None:continue
                loss,_=teacher_loss(model,ex)
                if not torch.isfinite(loss):raise ValueError('Non-finite teacher loss')
                (loss/8).backward();accum+=1;running.append(float(loss.detach()))
                if accum==8:
                    nn.utils.clip_grad_norm_(adapters.params(role),1.);optimizer.step();optimizer.zero_grad();accum=0
                if (i+1)%64==0:print(f'{role}: supervised epoch {epoch+1}/{epochs}, {i+1}/{len(order)} examples',flush=True)
            if accum:
                for p in adapters.params(role):
                    if p.grad is not None:p.grad.mul_(8/accum)
                nn.utils.clip_grad_norm_(adapters.params(role),1.);optimizer.step();optimizer.zero_grad()
            val=evaluate_teacher(model,adapters,role,examples['validation'])
            logs[role]['epochs'].append(dict(epoch=epoch+1,train_nll=float(np.mean(running)),validation=val))
            if val['nll']<best:
                best=val['nll'];logs[role]['selected_epoch']=epoch+1
                for k,v in adapters.state().items():
                    if f'.{role}.' in k:selected[k]=v
            torch.save(dict(format_version=VERSION,adapters=adapters.state(),selected=selected,logs=logs,initial=initial,epochs=epochs,examples=len(examples['train']),
                            optimizers={r:o.state_dict() for r,o in optimizers.items()}),OUT/'sft_progress.pt')
            print(f'{role}: epoch {epoch+1} validation NLL {val["nll"]:.4f}; best {best:.4f}',flush=True)
        # Restore this instrument's best epoch, including the untrained fallback.
        current=adapters.state()
        for k in current:
            if f'.{role}.' in k:current[k]=selected[k]
        adapters.load(current)
    adapters.load(selected)
    for role in nt.ROLES:
        logs[role]['selected_validation']=evaluate_teacher(model,adapters,role,examples['validation'])
        logs[role]['selected_without_partners']=evaluate_teacher(model,adapters,role,examples['validation'],shared=False)
    return logs,initial


def rollout(model,adapters,agents,critic,brief,seed,order=None):
    order=list(nt.ROLES) if order is None else list(order)
    if sorted(order)!=sorted(nt.ROLES):raise ValueError('Each role must propose once')
    ledger=[];steps=[];trace=[];decisions=[]
    for beat in range(brief.bars*4):
        history=nt.visible(ledger,beat);state=nt.state_features(brief,beat,history)
        cs=critic_state(brief,beat,history)
        with torch.no_grad():value=float(critic(cs))
        actions={};records={}
        for role in order:
            adapters.activate(role)
            sampler=EnsembleSampler(model,role,seed+beat*1009+list(nt.ROLES).index(role)*100003,brief,beat,history)
            actions[role],records[role]=sampler.propose(agents[role],state)
            decisions.extend(dict(role=role,beat=beat,note=copy.deepcopy(n)) for n in actions[role])
            trace.append(dict(role=role,beat=beat,history_hash=hashlib.sha256(json.dumps(history).encode()).hexdigest(),
                              adapter=role,partner_notes=sum(n[3]!=nt.ROLES[role]['channel'] for n in history),
                              note_decisions=len(actions[role])))
        ledger=nt.commit(ledger,actions,beat,brief)
        score,parts=nt.reward(ledger,(beat+1)*480) if beat%4==3 else (0.,{})
        steps.append(dict(state=state,critic_state=cs,value=value,reward=score,components=parts,records=records))
    assert sorted(ledger)==sorted(d['note'] for d in decisions)
    return dict(notes=ledger,steps=steps,trace=trace,decisions=decisions,brief=nt.asdict(brief),seed=seed)


def main():
    global OUT
    p=argparse.ArgumentParser();p.add_argument('--examples',type=int,default=256);p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--ppo-updates',type=int,default=2);p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--output',type=Path,default=OUT);p.add_argument('--resume',action='store_true');args=p.parse_args()
    OUT=args.output.resolve()
    if args.prepare_only:ed.prepare();return
    if args.examples<12 or args.epochs<1 or args.ppo_updates<1:p.error('Invalid training size')
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'complete.pt').exists():raise ValueError('Completed experiment already exists; choose a new --output')
    if (OUT/'data_manifest.json').exists() and not args.resume:raise ValueError('Use --resume or a new --output to preserve the existing experiment')
    torch.set_num_threads(4);torch.manual_seed(20261103);random.seed(20261103);np.random.seed(20261103)
    examples,stats=make_examples(args.examples)
    manifest=dict(dataset_audit=str(ed.DATA/'audit.json'),dataset_audit_sha256=hashlib.sha256((ed.DATA/'audit.json').read_bytes()).hexdigest(),stats=stats,
        examples={s:[dict(song=e['song'],window_start=e['window_start'],beat=e['beat'],available_roles=e['available_roles'],projection=e['projection'],notes_sha256=hashlib.sha256(json.dumps(e['notes']).encode()).hexdigest()) for e in es] for s,es in examples.items()},
        proxies=False,missing_role_loss='Skipped, never treated as silence',
        source_projection='Prune to generation density/polyphony; highest upper-string voices, lowest monophonic cello with overlap shortening. No invented pitches/onsets.',
        teacher_forcing='Committed past partner notes, own earlier current-beat notes, exact sampler masks, no key prompt')
    manifest_path=OUT/'data_manifest.json'
    if args.resume and manifest_path.exists() and json.loads(manifest_path.read_text())!=manifest:raise ValueError('Dataset or examples differ from the saved run')
    manifest_path.write_text(json.dumps(manifest,indent=2))
    model=nt.load_model();adapters=InstrumentAdapters(model);base=adapters.base_hash()
    if args.resume and (OUT/'supervised.pt').exists():
        saved=torch.load(OUT/'supervised.pt',map_location='cpu',weights_only=False)
        if saved['epochs']!=args.epochs or saved['examples']!=args.examples:raise ValueError('Resume settings differ')
        adapters.load(saved['adapters']);logs=saved['logs'];initial=saved['initial']
        print('Resumed completed supervised stage',flush=True)
    else:
        logs,initial=supervised_train(model,adapters,examples,args.epochs,resume=args.resume)
    (OUT/'supervised_log.json').write_text(json.dumps(logs,indent=2))
    trained_adapters=adapters.state()
    torch.save(dict(format_version=VERSION,roles=nt.ROLES,adapters=trained_adapters,logs=logs,initial=initial,epochs=args.epochs,examples=args.examples,
                    model_revision='0f8f265d4330f4e46527ac2313200254c5757f5f'),OUT/'supervised.pt')
    agents={r:nt.Musician() for r in nt.ROLES};critic=SequenceCritic()
    opts={r:torch.optim.Adam(a.parameters(),lr=1e-4) for r,a in agents.items()};co=torch.optim.Adam(critic.parameters(),lr=3e-4)
    critic_before=nt.fingerprint(critic);rl_logs=[]
    if args.resume and (OUT/'latest.pt').exists():
        saved=torch.load(OUT/'latest.pt',map_location='cpu',weights_only=False)
        if saved['critic_type']!='sequence-gru-v1':raise ValueError('Wrong critic checkpoint')
        for r in nt.ROLES:agents[r].load_state_dict(saved['actors'][r]);opts[r].load_state_dict(saved['optimizers'][r])
        critic.load_state_dict(saved['critic']);co.load_state_dict(saved['critic_optimizer']);rl_logs=saved['logs']
        critic_before=saved['critic_before']
    for u in range(len(rl_logs),args.ppo_updates):
        eps=[]
        for i in range(4):
            eps.append(rollout(model,adapters,agents,critic,nt.Brief(4),6500001+u*4+i))
            print(f'Sequence-critic PPO {u+1}/{args.ppo_updates}, episode {i+1}/4',flush=True)
        rl_logs.append(nt.update(agents,critic,opts,co,eps))
        torch.save(dict(format_version=VERSION,roles=nt.ROLES,adapters=trained_adapters,
                        model_revision='0f8f265d4330f4e46527ac2313200254c5757f5f',sampler='neutral-key-ensemble-v1',
                        actors={r:a.state_dict() for r,a in agents.items()},critic=critic.state_dict(),
                        critic_type='sequence-gru-v1',critic_input=CRITIC_INPUT,optimizers={r:o.state_dict() for r,o in opts.items()},
                        critic_optimizer=co.state_dict(),critic_before=critic_before,completed_updates=u+1,logs=rl_logs),OUT/'latest.pt')
    # Separate seed; no listening sample is used to choose supervised epochs.
    checks={};provenance={}
    for name,policies in [('trained',agents),('supervised_only',{r:nt.Musician() for r in nt.ROLES}),('unadapted',{r:nt.Musician() for r in nt.ROLES})]:
        adapters.load(initial if name=='unadapted' else trained_adapters)
        ep=rollout(model,adapters,policies,critic,nt.Brief(24),9601001)
        checks[name]=nt.export(OUT,name,ep,stems=name=='trained')
        provenance[name]=dict(notes=len(ep['notes']),direct_decisions=len(ep['decisions']))
        print(f'Exported {name}',flush=True)
    adapters.load(trained_adapters)
    result=dict(format_version=VERSION,supervised_examples={s:len(v) for s,v in examples.items()},dataset_stats=stats,
        supervised_songs={s:len({w['song'] for w in ws}) for s,ws in examples.items()},
        supervised_songs_per_role={s:{r:len({w['song'] for w in ws if r in w['available_roles']}) for r in nt.ROLES} for s,ws in examples.items()},
        instrument_programs={r:sorted(v) for r,v in ed.PROGRAMS.items()},
        proxies_used=False,supervised=logs,adapter_parameter_changes={r:sum(float((v-initial[k]).float().square().sum()) for k,v in trained_adapters.items() if f'.{r}.' in k) for r in nt.ROLES},
        frozen_base_unchanged=adapters.base_hash()==base,sequence_critic_changed=nt.fingerprint(critic)!=critic_before,
        critic='Ordered preceding-eight-bar note context plus brief; heuristic-return value estimator, not musical-taste evaluator',
        ppo_updates=args.ppo_updates,ppo_episodes=args.ppo_updates*4,rl_logs=rl_logs,provenance=provenance,exports=checks,
        quality_improvement_established=False)
    assert result['frozen_base_unchanged'] and result['sequence_critic_changed']
    (OUT/'verification.json').write_text(json.dumps(result,indent=2))
    # Reuse the familiar player, with accurate methods and labels.
    nt.write_page(OUT);page=OUT/'listen.html';html=page.read_text(encoding='utf-8')
    html=html.replace('Three note-generating musicians','Supervised organ, strings and cello')
    html=html.replace('<a href="verification.json">Run verification</a>','<a href="verification.json">Run verification</a> · <a href="REPORT.md">Training report</a>')
    html=html.replace('<option value="initial">Matched untrained baseline</option>','<option value="supervised_only">Supervised adapters only</option><option value="unadapted">Before ensemble supervision</option>')
    html=html.replace('<option value="no_partner">Trained without partner observations</option>','')
    html=html.replace('This first joint-training result has','This supervised-plus-RL pilot has')
    html=html.replace('a central training-only critic','a training-only sequence critic reading the preceding eight bars')
    html=html.replace('After joint training','Supervised adapters + joint RL')
    html=html.replace('The three independently trained residual policies adapt a shared frozen pretrained MIDI model.',
        'Three separate supervised LoRA adapters learn real organ, upper-string and cello parts from aligned ensemble pairs; missing source instruments have no supervised loss. Independently trained residual policies then receive joint three-agent PPO updates. A sequence critic estimates the engineering reward during training.')
    page.write_text(html,encoding='utf-8')
    report=['# Supervised ensemble + sequence critic pilot','',
        'Each role has its own rank-8 LoRA adapter on the shared frozen MIDI backbone. Supervised labels are aligned real ensemble parts from Slakh MIDI, using instrument program identities (no proxies). Most examples are pairs, because full organ/strings/cello trios are too rare for separate training and validation. Missing instruments receive no supervised loss. Each role sees committed partner history and its own earlier notes in the current beat. Partners propose privately, then commit together.',
        '',f'Examples: {len(examples["train"])} training, {len(examples["validation"])} validation, with composition-level official split separation and canonical exact-duplicate exclusion. Validation chooses adapter epochs; this is not an independent test-set quality result. See data_manifest.json and assets/trio_ensemble/audit.json for actual coverage and attribution.',
        '', 'The teacher loss uses the same legal choices, temperature and exploration mixture as generation. Fixed track/channel/time tokens are excluded from loss. Source voices are pruned to the agents\' density/polyphony limits: higher string chord voices, lowest monophonic cello with overlap shortening. This preprocessing invents no pitches or onsets; source/retained note counts are recorded in data_manifest.json. The corpus has no reliable key labels, so this experiment omits the key-signature prompt during both training and generation. Timing is normalized to the generation grid and 100 BPM.',
        '', '| Instrument | Initial validation NLL | Selected validation NLL | Epoch |','|---|---:|---:|---:|']
    for r,l in logs.items():
        selected=l['initial_validation']['nll'] if not l['selected_epoch'] else l['epochs'][l['selected_epoch']-1]['validation']['nll']
        report.append(f'| {r} | {l["initial_validation"]["nll"]:.4f} | {selected:.4f} | {l["selected_epoch"]} |')
    report.extend(['','Partner-history ablation on the same validation targets (lower NLL is better):',
        '', '| Instrument | Selected, shared history | Selected, partners removed |','|---|---:|---:|'])
    for r,l in logs.items():
        report.append(f'| {r} | {l["selected_validation"]["nll"]:.4f} | {l["selected_without_partners"]["nll"]:.4f} |')
    report.append('\nThese ablations must be interpreted per instrument. Lower loss without partners is not evidence of better cooperation; lower loss with partners is only descriptive evidence on this small subset.')
    report.extend(['','The GRU critic sees up to 256 ordered notes from the preceding eight bars, including timing, duration, pitch, velocity and role, plus the brief. It is used only during training to estimate the existing heuristic reward. It is not a learned evaluator of human musical taste.',
        '',f'Joint PPO: {args.ppo_updates} updates, {args.ppo_updates*4} four-bar episodes. Only the separate residual actors and critic receive PPO gradients; supervised adapters stay frozen in this stage.',
        '', 'Listening files: trained_mix.mp3 and instrument stems, plus supervised_only_mix.mp3 without residual PPO adaptation and unadapted_mix.mp3 before ensemble supervision, all using the same seed. No prompts or test listens were used to select these performances. Every MIDI note is an agent decision; no supplied motif, chord progression, phrase bank or scripted arranger is used.',
        '', 'Limits: sparse cello data and no sufficiently sized full-trio supervised corpus; normalized MIDI rather than mastered audio; short training windows; longer musical form is not learned explicitly. Near-duplicate compositions and overlap with the backbone pretraining corpus have not been fully audited. Lower validation NLL means better imitation on this validation subset, not established listener preference or learned cooperation. Human quality improvement remains unproven.'])
    (OUT/'REPORT.md').write_text('\n'.join(report),encoding='utf-8')
    torch.save(dict(verification=result,complete=True),OUT/'complete.pt')
    print(json.dumps({k:v for k,v in result.items() if k not in ['supervised','rl_logs','exports']},indent=2),flush=True)


if __name__=='__main__':main()
