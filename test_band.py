import copy
import unittest
import torch
from band import (ROLES,Brief,BandEnv,Musician,TeamCritic,OBS,visible_history,
                  fingerprint,update,ensemble_reward)

class BandTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(2); torch.manual_seed(42)

    def test_causal_history_and_partner_removal(self):
        ledger=[['note',0,480,0,60,80],['note',480,480,1,36,80],['note',960,120,9,36,80]]
        self.assertEqual(len(visible_history(ledger,1,'piano')),1)
        self.assertEqual(len(visible_history(ledger,2,'piano',False)),1)
        self.assertFalse(any(n[1]>=960 for n in visible_history(ledger,2,'drums')))

    def test_simultaneous_commit_order(self):
        actions={'piano':[['note',0,480,0,60,80]],'bass':[['note',0,480,1,36,80]],'drums':[['note',0,120,9,36,80]]}
        a,b=BandEnv(),BandEnv(); a.reset(Brief(),3); b.reset(Brief(),3)
        a.step(actions); b.step(dict(reversed(list(actions.items()))))
        self.assertEqual(a.ledger,b.ledger)
        with self.assertRaises(ValueError): a.step({'piano':[]})

    def test_zero_residual_matches_reference_and_replay(self):
        actor=Musician(); obs=torch.randn(OBS); ref=torch.zeros(128); ref[[36,40,44]]=torch.tensor([.2,.3,.5])
        d=actor.distribution(obs,ref)
        torch.testing.assert_close(d.probs,ref)
        self.assertTrue(torch.isneginf(d.log_prob(torch.tensor(1))))
        self.assertEqual(float(d.log_prob(torch.tensor(40))),float(actor.distribution(obs,ref).log_prob(torch.tensor(40))))

    def test_invalid_joint_action_is_atomic(self):
        env=BandEnv(); env.reset(Brief(),1)
        actions=dict(piano=[['note',0,480,0,60,80]],bass=[['note',0,480,1,120,70]],drums=[])
        with self.assertRaises(ValueError): env.step(actions)
        self.assertEqual(env.ledger,[]); self.assertEqual(env.beat,0)

    def test_sustained_note_polyphony_is_enforced(self):
        env=BandEnv(); env.reset(Brief(),1)
        env.step(dict(piano=[],bass=[['note',0,960,1,36,70]],drums=[]))
        with self.assertRaises(ValueError):
            env.step(dict(piano=[],bass=[['note',480,480,1,40,70]],drums=[]))
        self.assertEqual(env.beat,1)

    def test_actor_parameters_are_independent(self):
        actors={r:Musician() for r in ROLES}; before={r:fingerprint(a) for r,a in actors.items()}
        optimizer=torch.optim.Adam(actors['piano'].parameters(),lr=.01)
        ref=torch.zeros(128); ref[:2]=.5
        loss=-actors['piano'].distribution(torch.randn(OBS),ref).log_prob(torch.tensor(0))
        loss.backward(); optimizer.step()
        self.assertNotEqual(before['piano'],fingerprint(actors['piano']))
        for r in ['bass','drums']: self.assertEqual(before[r],fingerprint(actors[r]))

    def test_joint_update_changes_every_actor(self):
        actors={r:Musician() for r in ROLES}; critic=TeamCritic()
        optimizers={r:torch.optim.Adam(a.parameters(),lr=.001) for r,a in actors.items()}
        before={r:fingerprint(a) for r,a in actors.items()}; episodes=[]
        for _ in range(2):
            steps=[]
            for t in range(4):
                records={}
                for r,a in actors.items():
                    obs=torch.randn(OBS); ref=torch.zeros(128); ref[:2]=.5
                    records[r]=[dict(observation=obs,reference=ref,action=0,logp=float(a.distribution(obs,ref).log_prob(torch.tensor(0)).detach()))]
                steps.append(dict(state=torch.randn(32),reward=float(t==3),value=0,records=records))
            episodes.append(dict(steps=steps))
        logs=update(actors,critic,optimizers,torch.optim.Adam(critic.parameters(),lr=.001),episodes)
        for r in ROLES:
            self.assertNotEqual(before[r],fingerprint(actors[r])); self.assertGreater(logs[r]['tokens'],0)

    def test_reward_rejects_silence_and_flooding(self):
        brief=Brief()
        coherent=[]
        for i,p in enumerate([60,64,67,64]):
            coherent += [['note',i*480,480,0,p,80],['note',i*480,480,1,36,70],['note',i*480,120,9,36,60]]
        silence,_=ensemble_reward([],0,1920,brief)
        good,_=ensemble_reward(coherent,0,1920,brief)
        flooded=coherent+[['note',i*30,30,0,48+i%40,100] for i in range(64)]
        bad,_=ensemble_reward(flooded,0,1920,brief)
        self.assertGreater(good,silence); self.assertGreater(good,bad)

    def test_checkpoint_selection_requires_score_and_all_musicians(self):
        from continue_band import choose_best
        best=dict(mean=.2,all_agents_present=True)
        self.assertFalse(choose_best(best,dict(mean=.1,all_agents_present=True)))
        self.assertFalse(choose_best(best,dict(mean=.8,all_agents_present=False)))
        self.assertFalse(choose_best(best,dict(mean=.2,all_agents_present=True)))
        self.assertTrue(choose_best(best,dict(mean=.3,all_agents_present=True)))
        self.assertTrue(choose_best(dict(mean=1,all_agents_present=False),dict(mean=.3,all_agents_present=True)))

if __name__=='__main__': unittest.main()
