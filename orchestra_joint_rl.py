"""19-role residual PPO over a frozen format-4 ensemble. Reward-gated research code.

The released package has no accepted musical-taste reward. --check reports that
block before loading the GPU. A validated TorchScript reward can be integrated
through the explicit, hash-bound contract below; no legacy proxy bypass exists.
"""
import os
os.environ.setdefault('USE_TF','0')
os.environ.setdefault('USE_FLAX','0')
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import signal
import time
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import orchestra_learning as l
import offline_ensemble as oe
import note_trio as nt
import band
import orchestra_rl_selection as selection
from ppo_adapter import gae
from train_offline_ensemble import atomic
from train_ensemble_repair import write_json

FORMAT='orchestra_joint_rl_v1'
REWARD_SCHEMA='orchestra-terminal-notes-v1'
TOKENS=128
EVENT_SIZE=25  # relative onset, duration, pitch, velocity, held, valid, 19 role IDs
CRITIC_SIZE=64+TOKENS*EVENT_SIZE+1
REWARD_LIMIT=4096


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reward_check(directory,source):
    """Acceptance is evidence from a separate experiment, never set by PPO."""
    directory=Path(directory);path=directory/'acceptance.json'
    if not path.is_file():raise RuntimeError('No reward acceptance record; RL remains blocked.')
    status=json.loads(path.read_text())
    required=('automated_gate_passed','reference_gate_passed','midi_preference_gate_passed',
              'temporal_gate_passed','human_preference_gate_passed','generated_domain_gate_passed','rl_training_allowed')
    if not all(status.get(k) is True for k in required):
        raise RuntimeError('Reward validation failed or human ratings are missing; no RL update is allowed.')
    if status.get('schema')!=REWARD_SCHEMA or status.get('roles')!=list(l.ROLES):
        raise RuntimeError('Reward was not validated for this 19-role representation.')
    model=directory/'reward.ts';protocol=directory/'protocol.json';evidence=directory/'validation_results.json'
    for file,key in [(model,'reward_sha256'),(protocol,'protocol_sha256'),(evidence,'validation_results_sha256')]:
        if not file.is_file() or sha(file)!=status.get(key):raise RuntimeError('Missing or changed reward artifact: '+file.name)
    if status.get('generator_sha256')!=sha(source):raise RuntimeError('Reward acceptance covers a different supervised generator.')
    if status.get('runner_sha256')!=sha(__file__):raise RuntimeError('Reward acceptance covers a different rollout/reward implementation.')
    return status


def reward_inputs(notes,bars,bpm):
    if len(notes)>REWARD_LIMIT:raise ValueError('Reward note limit exceeded; truncation would change the objective.')
    x=torch.zeros(1,REWARD_LIMIT,6);mask=torch.zeros(1,REWARD_LIMIT,dtype=torch.bool)
    for i,n in enumerate(sorted(notes,key=lambda n:(n[1],n[3],n[4]))):
        x[0,i]=torch.tensor([n[1]/(bars*1920),n[2]/3840,n[3]/18,n[4]/127,n[5]/127,(n[1]+n[2])/(bars*1920)])
        mask[0,i]=True
    return x,mask,torch.tensor([[bars/32,bpm/180,1.]])


class TerminalReward:
    def __init__(self,directory,source):
        self.acceptance=reward_check(directory,source)
        self.model=torch.jit.load(str(Path(directory)/'reward.ts'),map_location='cpu').eval()
        self.model.requires_grad_(False)
    @torch.no_grad()
    def __call__(self,notes,bars,bpm):
        value=self.model(*reward_inputs(notes,bars,bpm))
        if value.numel()!=1 or not torch.isfinite(value).all() or not -1<=float(value)<=1:
            raise ValueError('Reward must produce one finite, calibrated scalar in [-1,1].')
        return float(value)


def actor_state(history,role,beat,bars,bpm):
    if any(n[1]>=beat*480 for n in history):raise ValueError('Future/current joint proposals leaked into actor state')
    own=l.ROLES[role]['id'];x=torch.zeros(32)
    recent=[n for n in history if n[1]>=(beat-8)*480 or n[1]+n[2]>beat*480]
    for offset,notes in [(0,[n for n in recent if n[3]!=own]),(12,[n for n in recent if n[3]==own])]:
        total=sum(n[2] for n in notes)
        for n in notes:x[offset+n[4]%12]+=n[2]/max(1,total)
    x[24:28]=torch.tensor([beat/(bars*4),bpm/180,np.sin(beat*np.pi/2),np.cos(beat*np.pi/2)])
    x[28]=min(1,sum(n[3]==own for n in recent)/32)
    x[29]=min(1,sum(n[3]!=own for n in recent)/(32*18))
    x[30]=sum(n[3]==own and n[1]+n[2]>beat*480 for n in recent)/l.ROLES[role]['polyphony']
    x[31]=sum(n[3]!=own and n[1]+n[2]>beat*480 for n in recent)/64
    return x


def critic_state(history,beat,bars,bpm):
    if any(n[1]>=beat*480 for n in history):raise ValueError('Noncausal critic state')
    summary=torch.zeros(64);start=max(0,(beat-24)*480)
    recent=[n for n in history if n[1]>=start or n[1]+n[2]>beat*480]
    for n in recent:
        summary[n[4]%12]+=n[2];summary[12+n[3]]+=1
        summary[31+n[3]]+=float(n[1]+n[2]>beat*480)
    summary[:12]/=summary[:12].sum().clamp_min(1)
    summary[12:31]/=32;summary[31:50]/=4
    summary[50:54]=torch.tensor([beat/(bars*4),bpm/180,np.sin(beat*np.pi/2),np.cos(beat*np.pi/2)])
    events=torch.zeros(TOKENS,EVENT_SIZE)
    selected=sorted(recent,key=lambda n:(n[1],n[3],n[4]))[-TOKENS:]
    for i,n in enumerate(selected):
        events[i,:6]=torch.tensor([(n[1]-beat*480)/11520,n[2]/3840,n[4]/127,n[5]/127,float(n[1]+n[2]>beat*480),1.])
        events[i,6+n[3]]=1
    return torch.cat([summary,events.flatten(),torch.tensor([float(len(selected))])])


class OrchestraCritic(nn.Module):
    def __init__(self):
        super().__init__();self.event=nn.Sequential(nn.Linear(EVENT_SIZE,64),nn.Tanh())
        self.sequence=nn.GRU(64,96,batch_first=True);self.value=nn.Sequential(nn.Linear(160,96),nn.Tanh(),nn.Linear(96,1))
    def forward(self,x):
        x=x.to(next(self.parameters()).device)
        single=x.ndim==1
        if single:x=x[None]
        if x.shape[-1]!=CRITIC_SIZE:raise ValueError('Wrong critic representation version')
        seq=x[:,64:-1].reshape(-1,TOKENS,EVENT_SIZE)
        packed=pack_padded_sequence(self.event(seq),x[:,-1].long().clamp(1,TOKENS).cpu(),batch_first=True,enforce_sorted=False)
        _,h=self.sequence(packed);value=self.value(torch.cat([x[:,:64],h[-1]],-1)).squeeze(-1)
        return value[0] if single else value


def load_supervised(path):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved.get('format_version')!=4 or saved['roles']!=l.ROLES:raise ValueError('A format-4 19-role supervised checkpoint is required')
    model=nt.load_model();adapters=l.Adapters(model);adapters.load(saved['adapters'])
    planner=l.Planner().to(model.device);planner.load_state_dict(saved['planner']);planner.eval().requires_grad_(False)
    conditions=l.Conditions().to(model.device);conditions.load_state_dict(saved['conditions']);conditions.eval().requires_grad_(False)
    if adapters.base_hash()!=saved['base_sha256']:raise ValueError('Frozen backbone hash differs')
    model.eval().requires_grad_(False);model.net.gradient_checkpointing_disable()
    neutral={r:oe.Musician().eval().requires_grad_(False) for r in l.ROLES}
    return model,adapters,planner,conditions,neutral,saved


@torch.no_grad()
def rollout(model,adapters,actors,planner,conditions,neutral,critic,bars,seed,mode,calm,reward,bpm=70,order=None):
    order=list(order or l.ROLES)
    if sorted(order)!=sorted(l.ROLES):raise ValueError('Every role must propose exactly once')
    goal=torch.tensor(calm['style'] if calm else [.25,.5,.3,.5],device=model.device)
    brief=nt.Brief(bars,bpm,0,False);brief.calm=calm is not None
    if calm is not None and 'velocity_max' in calm:brief.velocity_max=calm['velocity_max']
    ledger=[];steps=[];decisions=[]
    for start in range(0,bars*4,8):
        plans=l.plan_sequence(planner,ledger,bars,list(l.ROLES),goal,calm)
        # Frozen supervised private proposals are announced, then final RL actions
        # commit synchronously. No actor observes its own future targets/proposals.
        drafts=l.private_draft(model,adapters,neutral,planner,conditions,ledger,start,bars,list(l.ROLES),goal,seed,calm,bpm) if mode=='revised' else []
        for beat in range(start,min(start+8,bars*4)):
            history=l.visible(ledger,beat);cs=critic_state(history,beat,bars,bpm)
            actions={};records={}
            for role in order:
                adapters.activate(l.ROLES[role]['family']);partner=[n for n in drafts if n[3]!=l.ROLES[role]['id']]
                delta=conditions(plans[beat//4],l.draft_features(partner,role,start).to(model.device),role,goal)
                sampler=l.Sampler(oe.ConditionedModel(model,delta),role,seed+beat*1009+l.ROLES[role]['id']*100003,brief,beat,history)
                notes,records[role]=sampler.propose(actors[role],actor_state(history,role,beat,bars,bpm))
                for n in notes:n[3]=l.ROLES[role]['id']
                actions[role]=notes;decisions.extend(dict(role=role,beat=beat,note=copy.deepcopy(n)) for n in notes)
            ledger=l.commit(ledger,actions,beat,bars)
            steps.append(dict(state=torch.zeros(32),critic_state=cs,value=float(critic(cs)),reward=0.,records=records))
    # Terminal complete-phrase score; the critic never receives future notes.
    steps[-1]['reward']=float(reward(ledger,bars,bpm))
    if not np.isfinite(steps[-1]['reward']) or not -1<=steps[-1]['reward']<=1:raise ValueError('Invalid reward')
    assert sorted(ledger)==sorted(d['note'] for d in decisions)
    return dict(notes=ledger,steps=steps,decisions=decisions,brief=nt.asdict(brief),seed=seed,mode=mode)


def replay_check(actors,episodes):
    maximum=0.
    for ep in episodes:
        for step in ep['steps']:
            for role,rows in step['records'].items():
                if not rows:continue
                with torch.no_grad():
                    device=next(actors[role].parameters()).device
                    d=actors[role].distribution(torch.stack([r['observation'] for r in rows]).to(device),torch.stack([r['reference'] for r in rows]).to(device))
                    err=(d.log_prob(torch.tensor([r['action'] for r in rows],device=device))-torch.tensor([r['logp'] for r in rows],device=device)).abs().max()
                maximum=max(maximum,float(err))
    if maximum>1e-4:raise ValueError('PPO old-policy log-probability replay mismatch')
    return maximum


def update(actors,critic,optimizers,critic_optimizer,episodes):
    # band.update groups all tokens of a role's beat into one clipped joint action,
    # shares team GAE across roles, and applies KL to the frozen note distribution.
    # Its loop uses policies.items(), not the legacy three-role list.
    from orchestra_ppo_update import update as device_update
    return device_update(actors,critic,optimizers,critic_optimizer,episodes)


def rng_state():return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),numpy=np.random.get_state(),python=random.getstate())
def restore_rng(s):
    torch.set_rng_state(s['torch']);torch.cuda.set_rng_state_all(s['cuda']);np.random.set_state(s['numpy']);random.setstate(s['python'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--reward',type=Path,default=Path('provenance/reward_validation'))
    p.add_argument('--output',type=Path,default=Path('runs/a100_joint_rl_v1'));p.add_argument('--check',action='store_true')
    p.add_argument('--rollout-smoke',action='store_true',help='Two-bar frozen rollout and replay only; no rewards learned and no optimizer updates')
    p.add_argument('--updates',type=int,default=24);p.add_argument('--episodes',type=int,default=8);p.add_argument('--bars',type=int,default=8)
    p.add_argument('--policy-device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--hours',type=float,default=2);p.add_argument('--seed',type=int,default=9942001)
    p.add_argument('--resume',action='store_true');p.add_argument('--cpu-threads',type=int,default=4)
    args=p.parse_args()
    if not 1<=args.updates<=512 or not 1<=args.episodes<=16 or not 2<=args.bars<=8 or args.hours<=0:p.error('Invalid bounded recipe')
    if args.rollout_smoke:
        torch.set_num_threads(args.cpu_threads)
        model,adapters,planner,conditions,neutral,supervised=load_supervised(args.source)
        actors={r:oe.Musician().eval() for r in l.ROLES};critic=OrchestraCritic().eval()
        frozen=band.fingerprint(model)
        episode=rollout(model,adapters,actors,planner,conditions,neutral,critic,2,args.seed,
            supervised.get('composition_mode','planned'),supervised.get('calm_profile'),lambda *_:0.)
        replay=replay_check(actors,[episode])
        if frozen!=band.fingerprint(model):raise ValueError('Smoke changed frozen weights')
        counts={r:sum(len(s['records'][r]) for s in episode['steps']) for r in l.ROLES}
        if not all(counts.values()):raise ValueError('A role had no sampled decisions')
        args.output.mkdir(parents=True,exist_ok=True)
        write_json(args.output/'rollout_smoke.json',dict(passed=True,no_optimizer_updates=True,
            reward_validation_bypassed_for_inference_only=True,source_sha256=sha(args.source),role_decisions=counts,
            replay_error=replay,notes_match_actions=True,frozen_unchanged=True,gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'))
        print('RL rollout smoke passed; musical reward remains unvalidated.');return
    try:acceptance=reward_check(args.reward,args.source)
    except (RuntimeError,OSError,ValueError) as error:
        if args.check:print(json.dumps(dict(rl_allowed=False,reason=str(error))));return
        p.error(str(error))
    if args.check:print(json.dumps(dict(rl_allowed=True,acceptance=acceptance)));return
    torch.set_num_threads(args.cpu_threads);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    reward=TerminalReward(args.reward,args.source)
    root=args.output;root.mkdir(parents=True,exist_ok=True)
    if (root/'progress.pt').exists() and not args.resume:raise ValueError('Use a new output directory or --resume')
    model,adapters,planner,conditions,neutral,supervised=load_supervised(args.source)
    frozen=dict(model=band.fingerprint(model),planner=band.fingerprint(planner),conditions=band.fingerprint(conditions))
    policy_device='cuda' if args.policy_device=='auto' and torch.cuda.is_available() else 'cpu' if args.policy_device=='auto' else args.policy_device
    actors={r:oe.Musician().to(policy_device).eval() for r in l.ROLES};critic=OrchestraCritic().to(policy_device)
    opts={r:torch.optim.Adam(a.parameters(),lr=3e-5) for r,a in actors.items()};co=torch.optim.Adam(critic.parameters(),lr=1e-4)
    config=dict(source_sha256=sha(args.source),reward_sha256=acceptance['reward_sha256'],runner_sha256=sha(__file__),
                optimizer_sha256=sha(Path(__file__).with_name('orchestra_ppo_update.py')),
                updates=args.updates,episodes=args.episodes,bars=args.bars,seed=args.seed,roles=list(l.ROLES),algorithm='19 residual actors; centralized sequence critic; shared team GAE; clipped PPO; reference KL')
    state=dict(step=0,elapsed=0.,complete=False,status='Ready',updates=[],quality='unvalidated candidate; listening required')
    if args.resume:
        saved=torch.load(root/'progress.pt',map_location='cpu',weights_only=False)
        if saved['config']!=config or saved['format_version']!=FORMAT:raise ValueError('Resume config/source/reward mismatch')
        for r,a in actors.items():a.load_state_dict(saved['actors'][r]);opts[r].load_state_dict(saved['optimizers'][r])
        critic.load_state_dict(saved['critic']);co.load_state_dict(saved['critic_optimizer']);state=saved['state'];restore_rng(saved['rng'])
    started=time.monotonic();prior=state['elapsed'];stop=[False]
    def request_stop(*_):stop[0]=True
    signal.signal(signal.SIGINT,request_stop);signal.signal(signal.SIGTERM,request_stop)
    def save():
        state['elapsed']=prior+time.monotonic()-started
        atomic(root/'progress.pt',dict(format_version=FORMAT,config=config,actors={r:a.state_dict() for r,a in actors.items()},
            critic=critic.state_dict(),optimizers={r:o.state_dict() for r,o in opts.items()},critic_optimizer=co.state_dict(),
            state=state,rng=rng_state(),supervised_sha256=config['source_sha256']))
        write_json(root/'training_log.json',state)
    def generate(seed):return rollout(model,adapters,actors,planner,conditions,neutral,critic,args.bars,seed,
        supervised.get('composition_mode','planned'),supervised.get('calm_profile'),reward)
    def evaluate(seeds):
        result=[]
        for seed in seeds:
            if stop[0] or prior+time.monotonic()-started>=args.hours*3600:raise TimeoutError('Evaluation paused at episode boundary; resume to finish')
            result.append(selection.measurements(generate(seed)))
        return result
    def selected_bundle():
        saved=torch.load(root/'progress.pt',map_location='cpu',weights_only=False)
        atomic(root/'selected.pt',saved)
    # Each saved progress file is a complete optimizer update. If killed during
    # rollout/update, resume from the prior atomic checkpoint and replay its RNG.
    save()
    try:
        if 'validation_baseline' not in state:
            state['validation_baseline']=evaluate(selection.VALIDATION_SEEDS)
            state['best_mean']=float(np.mean([r['reward'] for r in state['validation_baseline']]))
            state['selected_step']=0;state['validated_step']=0;save();selected_bundle()
        for i in range(state['step'],args.updates):
            if state['step'] and state['step']%4==0 and state.get('validated_step')!=state['step']:
                candidate=evaluate(selection.VALIDATION_SEEDS)
                verdict=selection.select(candidate,state['validation_baseline'],state['best_mean'])
                state.setdefault('validations',[]).append(dict(step=state['step'],results=candidate,**verdict))
                state['validated_step']=state['step']
                if verdict['passed']:state.update(best_mean=verdict['mean_reward'],selected_step=state['step'])
                save()
                if verdict['passed']:selected_bundle()
            if stop[0] or prior+time.monotonic()-started>=args.hours*3600:break
            tick=time.monotonic();episodes=[]
            for j in range(args.episodes):
                episodes.append(generate(args.seed+i*args.episodes+j))
                if stop[0] or prior+time.monotonic()-started>=args.hours*3600:break
            if len(episodes)!=args.episodes or stop[0] or prior+time.monotonic()-started>=args.hours*3600:break
            replay=replay_check(actors,episodes)
            losses=update(actors,critic,opts,co,episodes)
            state['step']=i+1;state['status']='Training joint RL'
            row=dict(step=i+1,reward=float(np.mean([e['steps'][-1]['reward'] for e in episodes])),losses=losses,replay_error=replay,seconds=time.monotonic()-tick)
            state['updates'].append(row);save();print(json.dumps(row),flush=True)
        if state['step']==args.updates and state.get('validated_step')!=state['step']:
            candidate=evaluate(selection.VALIDATION_SEEDS);verdict=selection.select(candidate,state['validation_baseline'],state['best_mean'])
            state.setdefault('validations',[]).append(dict(step=state['step'],results=candidate,**verdict));state['validated_step']=state['step']
            if verdict['passed']:state.update(best_mean=verdict['mean_reward'],selected_step=state['step'])
            save()
            if verdict['passed']:selected_bundle()
        state['complete']=state['step']==args.updates;state['status']='Complete: validation selection finished; listening still required' if state['complete'] else 'Stopped at complete update'
        if frozen!=dict(model=band.fingerprint(model),planner=band.fingerprint(planner),conditions=band.fingerprint(conditions)):raise ValueError('Frozen supervised weights changed')
        save()
        # No automatic promotion by training reward. Preserve the supervised
        # checkpoint until composition-separated probes and listening favor PPO.
    except TimeoutError as error:
        state['status']=str(error);save();print(str(error),flush=True)
    except BaseException:
        # Never overwrite a good progress file with a partially updated actor set.
        raise


if __name__=='__main__':main()
