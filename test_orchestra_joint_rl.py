import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
import orchestra_joint_rl as rl


class OrchestraRLTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_failed_reward_blocks_before_gpu_loading(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d,'acceptance.json').write_text(json.dumps({'rl_training_allowed':False}))
            with self.assertRaisesRegex(RuntimeError,'validation failed'):rl.reward_check(d,'not-a-checkpoint')

    def test_flags_alone_cannot_enable_reward(self):
        keys=['automated_gate_passed','reference_gate_passed','midi_preference_gate_passed','temporal_gate_passed',
              'human_preference_gate_passed','generated_domain_gate_passed','rl_training_allowed']
        with tempfile.TemporaryDirectory() as d:
            Path(d,'acceptance.json').write_text(json.dumps({**dict.fromkeys(keys,True),'schema':rl.REWARD_SCHEMA,'roles':list(rl.l.ROLES)}))
            with self.assertRaisesRegex(RuntimeError,'artifact'):rl.reward_check(d,'not-a-checkpoint')

    def test_critic_represents_last_role_and_sustains(self):
        role=list(rl.l.ROLES)[-1];idx=rl.l.ROLES[role]['id']
        notes=[['note',0,3840,idx,60,48]]
        state=rl.critic_state(notes,4,8,70)
        events=state[64:-1].reshape(rl.TOKENS,rl.EVENT_SIZE)
        self.assertEqual(float(events[0,6+idx]),1.)
        self.assertEqual(float(events[0,4]),1.)
        self.assertEqual(float(state[31+idx]),.25)
        self.assertTrue(torch.isfinite(rl.OrchestraCritic()(state)))
        with self.assertRaises(ValueError):rl.critic_state(notes,0,8,70)

    def test_all_nineteen_actors_update_and_replay(self):
        torch.manual_seed(19)
        actors={r:rl.oe.Musician() for r in rl.l.ROLES};critic=rl.OrchestraCritic();steps=[]
        for beat in range(4):
            records={}
            for role,a in actors.items():
                obs=torch.randn(1072);ref=torch.zeros(129);ref[:2]=.5;action=beat%2
                with torch.no_grad():logp=float(a.distribution(obs,ref).log_prob(torch.tensor(action)))
                records[role]=[dict(observation=obs,reference=ref,action=action,logp=logp)]
            cs=rl.critic_state([],beat,2,70)
            with torch.no_grad():value=float(critic(cs))
            steps.append(dict(state=torch.zeros(32),critic_state=cs,value=value,reward=float(beat==3),records=records))
        eps=[dict(steps=steps)];self.assertLess(rl.replay_check(actors,eps),1e-5)
        before={r:rl.band.fingerprint(a) for r,a in actors.items()}
        opts={r:torch.optim.Adam(a.parameters(),lr=3e-5) for r,a in actors.items()}
        logs=rl.update(actors,critic,opts,torch.optim.Adam(critic.parameters(),lr=1e-4),eps)
        self.assertTrue(all(before[r]!=rl.band.fingerprint(a) for r,a in actors.items()))
        self.assertTrue(all(torch.isfinite(torch.tensor(logs[r]['loss'])) for r in actors))

    def test_joint_commit_order_and_no_current_observation(self):
        class Model:device='cpu'
        class Adapters:
            def activate(self,*_):pass
        class Conditions:
            def __call__(self,*_):return torch.zeros(1024)
        class Critic:
            def __call__(self,*_):return torch.tensor(0.)
        observed=[]
        class Sampler:
            def __init__(self,model,role,seed,brief,beat,history):
                self.role,self.beat=role,beat;observed.append((beat,copy.deepcopy(history)))
            def propose(self,*_):
                s=rl.l.ROLES[self.role]
                return [['note',self.beat*480,120,s['id'],s['pitches'][0],48]],[]
        agents={r:None for r in rl.l.ROLES}
        with patch.object(rl.l,'Sampler',Sampler),patch.object(rl.l,'plan_sequence',return_value=torch.zeros(2,128)):
            args=[Model(),Adapters(),agents,None,Conditions(),agents,Critic(),2,1,'planned',None,lambda *_:.25]
            a=rl.rollout(*args);b=rl.rollout(*args,order=list(reversed(agents)))
        self.assertEqual(a['notes'],b['notes']);self.assertEqual(len(a['notes']),19*8)
        self.assertTrue(all(n[1]<beat*480 for beat,history in observed for n in history))
        self.assertEqual(a['steps'][-1]['reward'],.25)
        self.assertTrue(all(s['reward']==0 for s in a['steps'][:-1]))

    def test_reward_keeps_rests_and_rejects_truncation(self):
        x,mask,brief=rl.reward_inputs([],8,70)
        self.assertFalse(mask.any());self.assertEqual(tuple(x.shape),(1,4096,6))
        with self.assertRaises(ValueError):rl.reward_inputs([['note',0,60,0,60,48]]*4097,8,70)


if __name__=='__main__':unittest.main()
