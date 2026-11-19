import unittest
import copy
from unittest.mock import patch
import torch
import band
import joint_ensemble_rl as rl
import offline_ensemble as oe
from sequence_critic import SequenceCritic,INPUT


class JointRLTests(unittest.TestCase):
    def test_team_reward_penalizes_identical_unison(self):
        notes=[['note',0,480,0,60,64],['note',0,480,1,60,64]]
        _,parts,_=rl.score(notes,1920)
        self.assertEqual(parts['exact_unison'],1)
        _,parts,_=rl.score([notes[0],['note',0,480,1,67,64]],1920)
        self.assertEqual(parts['exact_unison'],0)

    def test_selection_rejects_sacrificing_one_agent(self):
        initial=dict(team=.5,roles=dict(organ=.5,strings=.5,cello=.5))
        candidate=dict(team=.6,roles=dict(organ=.6,strings=.6,cello=.4))
        self.assertFalse(rl.eligible(candidate,initial,.5))
        candidate['roles']['cello']=.5
        self.assertTrue(rl.eligible(candidate,initial,.5))

    def test_ppo_updates_all_three_policies_from_rollouts(self):
        torch.manual_seed(1)
        agents={r:oe.Musician() for r in rl.er.ROLES};critic=SequenceCritic()
        steps=[]
        for beat in range(4):
            records={}
            for role,actor in agents.items():
                observation=torch.randn(1072);reference=torch.zeros(129);reference[:2]=.5
                with torch.no_grad():dist=actor.distribution(observation,reference)
                records[role]=[dict(observation=observation,reference=reference,action=beat%2,
                                    logp=float(dist.log_prob(torch.tensor(beat%2))))]
            cs=torch.zeros(INPUT)
            with torch.no_grad():value=float(critic(cs))
            steps.append(dict(state=torch.zeros(32),critic_state=cs,value=value,reward=float(beat==3),records=records))
        ep=dict(steps=steps);self.assertLess(rl.replay_check(agents,ep),1e-5)
        before={r:band.fingerprint(a) for r,a in agents.items()}
        opts={r:torch.optim.Adam(a.parameters(),lr=3e-5) for r,a in agents.items()}
        logs=band.update(agents,critic,opts,torch.optim.Adam(critic.parameters()),[ep])
        self.assertTrue(all(before[r]!=band.fingerprint(a) for r,a in agents.items()))
        self.assertTrue(all(torch.isfinite(torch.tensor(logs[r]['loss'])) for r in agents))

    def test_simultaneous_commit_and_order_independence(self):
        class Model:device='cpu'
        class Adapters:
            def activate(self,role):pass
        class Conditions:
            def __call__(self,*args):return torch.zeros(1024)
        class Critic:
            def __call__(self,state):return torch.tensor(0.)
        observed=[]
        class Sampler:
            def __init__(self,model,role,seed,brief,beat,history):
                self.role,self.beat=role,beat
                self.history=copy.deepcopy(history)
                observed.append((beat,role,copy.deepcopy(history)))
            def propose(self,actor,state):
                index=rl.er.ROLES.index(self.role)
                return [['note',self.beat*480,120,index,[60,72,48][index],64]],[]
        with patch.object(rl.oe,'Sampler',Sampler),patch.object(rl.er,'plan_segment',return_value=([torch.zeros(48)]*2,None,None)):
            agents={r:None for r in rl.er.ROLES}
            a=rl.rollout(Model(),Adapters(),agents,None,Conditions(),Critic(),2,1)
            b=rl.rollout(Model(),Adapters(),agents,None,Conditions(),Critic(),2,1,order=list(reversed(rl.er.ROLES)))
        self.assertEqual(a['notes'],b['notes'])
        self.assertEqual(len(a['notes']),24)
        self.assertTrue(all(n[1]<beat*480 for beat,role,history in observed for n in history))
        self.assertEqual(len(a['decisions']),len(a['notes']))


if __name__=='__main__':unittest.main()
