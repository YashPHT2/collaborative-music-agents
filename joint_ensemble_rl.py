"""Three on-policy note actors with a centralized sequence critic (joint PPO).

The v3 supervised adapters/planner stay frozen. Independent trainable residual
actors select event/rest, onset, pitch, duration and velocity at every beat.
Shared rewards are transparent engineering proxies, not a learned taste model.
"""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
import band
import ensemble_repair as er
import note_trio as nt
import offline_ensemble as oe
from partner_response import ensemble_score
from sequence_critic import SequenceCritic,critic_state
from train_ensemble_repair import OUT as SUPERVISED,atomic,write_json

OUT=nt.ROOT/'outputs'/'joint_ensemble_rl_v1'
VALIDATION_SEEDS=[984100,984101,984102]
TEST_SEEDS=[985100,985101,985102]


def score(notes,end):
    team,parts=ensemble_score(notes,end)
    # Repeated same-pitch notes and exact unison between roles are explicit
    # degeneration diagnostics. Rest itself is never an invalid action.
    recent=[n for n in notes if end-1920<=n[1]<end]
    attacks={r:[n for n in recent if n[3]==i] for i,r in enumerate(er.ROLES)}
    pairs=[(a,b) for i,a in enumerate(recent) for b in recent[i+1:] if a[3]!=b[3] and a[1]==b[1]]
    unison=sum(a[4]==b[4] for a,b in pairs)/max(1,len(pairs))
    parts['exact_unison']=unison
    parts['proxy_score']=team
    value=float(np.clip(team-.1*unison,-1,1))
    # Per-role diagnostics used to prevent sacrificing one instrument to improve
    # the average; these are not presented as individual causal credit estimates.
    per_role={r:ensemble_score(notes,end,[r])[0] for r in er.ROLES}
    return value,parts,per_role


@torch.no_grad()
def rollout(model,adapters,agents,planner,conditions,critic,bars,seed,context='correct',order=None):
    if context not in ('correct','missing','mismatched'):raise ValueError('Unknown context')
    order=order or er.ROLES
    if sorted(order)!=sorted(er.ROLES):raise ValueError('Each role must act once')
    if getattr(getattr(planner,'head',None),'out_features',None)==32:
        return offline_rollout(model,adapters,agents,planner,conditions,critic,bars,seed,context,order)
    brief=nt.Brief(bars);ledger=[];steps=[];trace=[];decisions=[];plans=[];previous=hidden=None
    with oe.limits():
        for beat in range(bars*4):
            true_history=nt.visible(ledger,beat)
            if beat%8==0:
                plan,previous,hidden=er.plan_segment(planner,true_history,beat,bars,previous,hidden)
                plans.extend(plan)
            cs=critic_state(brief,beat,true_history)
            value=float(critic(cs));actions={};records={}
            for role in order:
                history=copy.deepcopy(true_history);channel=er.ROLES.index(role)
                if context=='missing':history=[n for n in history if n[3]==channel]
                elif context=='mismatched':
                    for n in history:
                        if n[3]!=channel:n[4]=min(127,n[4]+6)
                # Online simultaneous play: no uncommitted partner future notes.
                # Supervised two-pass revision is not replayed as on-policy data.
                adapters.activate(role)
                delta=conditions(plans[beat//4],torch.zeros(1,8,device=model.device),role)
                sampler=oe.Sampler(oe.ConditionedModel(model,delta),role,seed+beat*1009+channel*100003,brief,beat,history)
                actions[role],records[role]=sampler.propose(agents[role],nt.state_features(brief,beat,history))
                trace.append(dict(role=role,beat=beat,context=context,history_sha256=hashlib.sha256(json.dumps(history).encode()).hexdigest(),
                                  partner_notes=sum(n[3]!=channel for n in history)))
            ledger=nt.commit(ledger,actions,beat,brief)
            for role,notes in actions.items():
                decisions.extend(dict(role=role,beat=beat,note=n,origin='joint_ppo_policy') for n in notes)
            reward,parts,roles=score(ledger,(beat+1)*480) if beat%4==3 else (0.,{}, {})
            steps.append(dict(state=nt.state_features(brief,beat,true_history),critic_state=cs,value=value,
                              reward=reward,components=parts,role_scores=roles,records=records))
    assert sorted(ledger)==sorted(d['note'] for d in decisions)
    return dict(notes=ledger,steps=steps,trace=trace,decisions=decisions,brief=nt.asdict(brief),seed=seed,context=context,
                plans=[p.cpu().tolist() for p in plans])


@torch.no_grad()
def offline_rollout(model,adapters,agents,planner,conditions,critic,bars,seed,context='correct',order=None):
    """Exact earlier v2 drafting/revision schedule, with final-policy PPO records."""
    ep=oe.compose(model,adapters,agents,planner,conditions,bars,seed,'revised',order=order,partner_mode=context,collect_training=True)
    records=ep.pop('policy_records');ep['steps']=[];brief=nt.Brief(bars)
    for beat in range(bars*4):
        history=nt.visible(ep['notes'],beat);cs=critic_state(brief,beat,history)
        reward,parts,roles=score(ep['notes'],(beat+1)*480) if beat%4==3 else (0.,{}, {})
        ep['steps'].append(dict(state=nt.state_features(brief,beat,history),critic_state=cs,value=float(critic(cs)),
                                reward=reward,components=parts,role_scores=roles,records=records[beat]))
    return ep


def metrics(episodes):
    values=[s for ep in episodes for s in ep['steps'] if s['role_scores']]
    return dict(team=float(np.mean([s['reward'] for s in values])),
                roles={r:float(np.mean([s['role_scores'][r] for s in values])) for r in er.ROLES},
                participation=[er.participation(ep['notes'],ep['brief']['bars']) for ep in episodes])


def eligible(candidate,baseline,best):
    return candidate['team']>best and all(candidate['roles'][r]>=baseline['roles'][r]-.02 for r in er.ROLES)


def replay_check(agents,ep):
    largest=0.
    for step in ep['steps']:
        for role,records in step['records'].items():
            if not records:continue
            obs=torch.stack([r['observation'] for r in records]);refs=torch.stack([r['reference'] for r in records])
            actions=torch.tensor([r['action'] for r in records])
            with torch.no_grad():actual=agents[role].distribution(obs,refs).log_prob(actions)
            largest=max(largest,float((actual-torch.tensor([r['logp'] for r in records])).abs().max()))
    if largest>1e-4:raise ValueError('On-policy log probability replay failed')
    return largest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=SUPERVISED/'latest.pt');parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--updates',type=int,default=48);parser.add_argument('--hours',type=float,default=1.)
    parser.add_argument('--legacy-proxy-experiment',action='store_true',
                        help='Explicitly reproduce the old engineering-proxy experiment; this is not the new preference reward.')
    parser.add_argument('--resume',action='store_true');parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    if not 1<=args.updates<=128 or not 0<args.hours<=1:parser.error('updates 1..128; hours >0 and <=1')
    # The user requested reward validation BEFORE any further policy training.
    # A newly trained failure detector must not silently replace musical taste,
    # and the historical engineering proxy must not masquerade as the new reward.
    if not args.legacy_proxy_experiment:
        from preference_reward import require_validated_reward
        try:require_validated_reward(nt.ROOT/'outputs'/'reward_validation_v2')
        except RuntimeError as error:parser.error(str(error))
        parser.error('Reward validation is complete, but no validated scalar blend has been enabled. Review reward_validation_v2 before policy training.')
    root=args.output;root.mkdir(parents=True,exist_ok=True)
    if (root/'progress.pt').exists() and not args.resume:raise ValueError('Use --resume or a new output')
    torch.set_num_threads(4);torch.manual_seed(984001)
    source=torch.load(args.source,map_location='cpu',weights_only=False)
    loader=oe.load if source.get('format_version')==2 else er.load
    model,adapters,agents,planner,conditions=loader(args.source)
    planner.eval();conditions.eval();planner.requires_grad_(False);conditions.requires_grad_(False)
    base=adapters.base_hash();adapter_hash=band.fingerprint(adapters.model) # includes frozen adapters
    critic=SequenceCritic();opts={r:torch.optim.Adam(agents[r].parameters(),lr=3e-5) for r in er.ROLES}
    critic_opt=torch.optim.Adam(critic.parameters(),lr=1e-4)
    config=dict(source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),updates=args.updates,hours=args.hours,
                algorithm='joint clipped PPO, shared team GAE, centralized sequence critic',trainable='three independent residual note actors and critic',
                reward='ensemble_score minus 0.1 exact cross-role unison; engineering proxy',smoke=args.smoke)
    state=dict(stage='joint_rl',step=0,total=args.updates,elapsed=0.,complete=False,logs=[],best=None)
    def bundle():
        return dict(format_version='joint_rl_v1',source=str(args.source.resolve()),config=config,
                    policies={r:agents[r].state_dict() for r in er.ROLES},critic=critic.state_dict(),source_sha256=config['source_sha256'])
    if args.resume:
        saved=torch.load(root/'progress.pt',map_location='cpu',weights_only=False)
        if saved['config']!=config:raise ValueError('Resume recipe differs')
        for r in er.ROLES:agents[r].load_state_dict(saved['policies'][r]);opts[r].load_state_dict(saved['optimizers'][r])
        critic.load_state_dict(saved['critic']);critic_opt.load_state_dict(saved['critic_optimizer']);state=saved['state']
        torch.set_rng_state(saved['rng'])
    started=time.monotonic();prior=state['elapsed'];deadline=args.hours*3600
    def elapsed():return prior+time.monotonic()-started
    def save():
        state['elapsed']=elapsed()
        atomic(root/'progress.pt',{**bundle(),'state':state,'optimizers':{r:o.state_dict() for r,o in opts.items()},
                                  'critic_optimizer':critic_opt.state_dict(),'rng':torch.get_rng_state()})
        write_json(root/'training_log.json',state)
    def episodes(seeds,bars=4,context='correct'):
        result=[]
        for seed in seeds:
            if elapsed()>=deadline:raise TimeoutError('RL time budget exhausted')
            result.append(rollout(model,adapters,agents,planner,conditions,critic,bars,seed,context))
        return result
    try:
        if state['best'] is None:
            print('RL baseline validation',flush=True)
            state['baseline']=metrics(episodes(VALIDATION_SEEDS[:1] if args.smoke else VALIDATION_SEEDS,2 if args.smoke else 4))
            state['best']=state['baseline']['team'];atomic(root/'initial.pt',bundle());atomic(root/'selected.pt',bundle());save()
        for update_index in range(state['step'],args.updates):
            # Reserve the final quarter for test interventions and audio, within one hour.
            if elapsed()>=deadline*.75:break
            batch=episodes([9841000+update_index*2+i for i in range(1 if args.smoke else 2)],2 if args.smoke else 4)
            replay=replay_check(agents,batch[0]);before={r:band.fingerprint(agents[r]) for r in er.ROLES}
            losses=band.update(agents,critic,opts,critic_opt,batch)
            changed={r:before[r]!=band.fingerprint(agents[r]) for r in er.ROLES}
            if not all(changed.values()):raise ValueError('An actor received no policy update')
            state['step']=update_index+1
            row=dict(step=state['step'],rollout=metrics(batch),losses=losses,actors_changed=changed,replay_error=replay)
            if state['step']%4==0 or args.smoke or state['step']==args.updates:
                val=metrics(episodes(VALIDATION_SEEDS[:1] if args.smoke else VALIDATION_SEEDS,2 if args.smoke else 4))
                row['validation']=val;row['eligible']=eligible(val,state['baseline'],state['best'])
                if row['eligible']:
                    state['best']=val['team'];state['selected_step']=state['step'];atomic(root/'selected.pt',bundle())
            state['logs'].append(row);save()
            print(f'joint_rl {state["step"]}/{args.updates}: team_reward={row["rollout"]["team"]:.4f} actor_losses='+str({r:losses[r]['loss'] for r in er.ROLES}),flush=True)
        atomic(root/'latest.pt',bundle())
        selected=torch.load(root/'selected.pt',map_location='cpu',weights_only=False)
        for r in er.ROLES:agents[r].load_state_dict(selected['policies'][r])
        selected_actors={r:copy.deepcopy(agents[r].state_dict()) for r in er.ROLES}
        evaluation={}
        for name in ('supervised','rl_selected'):
            if name=='supervised':
                for actor in agents.values():
                    torch.nn.init.zeros_(actor.head.weight);torch.nn.init.zeros_(actor.head.bias)
            else:
                for r in er.ROLES:agents[r].load_state_dict(selected_actors[r])
            evaluation[name]={}
            for context in ('correct','missing','mismatched'):
                eps=episodes(TEST_SEEDS[:1] if args.smoke else TEST_SEEDS,2 if args.smoke else 8,context)
                evaluation[name][context]=metrics(eps)
                if context=='correct':
                    nt.export(root,name,eps[0],allow_silent_stems=True)
        if adapters.base_hash()!=base or band.fingerprint(adapters.model)!=adapter_hash:raise ValueError('Frozen source weights changed')
        write_json(root/'evaluation.json',dict(results=evaluation,selected_step=state.get('selected_step',0),
                   quality='pending listening',interpretation='Proxy reward and missing/mismatched interventions do not establish human preference or causal credit assignment.'))
        (root/'listen.html').write_text('<!doctype html><meta charset="utf-8"><title>Joint RL trio</title><body style="font:20px system-ui;max-width:850px;margin:40px auto;background:#101724;color:white"><h1>Joint multi-agent RL</h1><p>Same held-out seed, renderer and length. All notes are policy actions. Listener preference remains untested.</p>'+''.join(f'<h2>{label}</h2>'+''.join(f'<p>{stem}</p><audio controls src="{name}_{stem}.mp3"></audio>' for stem in ['mix',*er.ROLES]) for name,label in [('supervised','Supervised foundation'),('rl_selected','Selected joint PPO policy')])+'<p><a href="evaluation.json">Evaluation and selection</a></p>',encoding='utf-8')
        state['complete']=True;save()
        print('Joint RL completed. Selected update:',state.get('selected_step',0),flush=True)
    except TimeoutError as error:
        state['budget_stopped']=True;state['status']=str(error);save()
        atomic(root/'latest.pt',bundle());print(str(error),flush=True)
    except BaseException:
        save();raise


if __name__=='__main__':main()
