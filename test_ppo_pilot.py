import unittest

import torch

from ppo_adapter import ResidualPolicy, gae, ppo_update, reference_kl, features
from pilot_environment import synthetic_episode, window_reward


class PPOTests(unittest.TestCase):
    def test_initial_actor_matches_reference_with_masked_actions(self):
        torch.manual_seed(1)
        policy = ResidualPolicy()
        ref = torch.zeros(2,33)
        ref[:,:2] = torch.tensor([0.8,0.2])
        distribution,_ = policy.distribution(torch.zeros(2,64),ref)
        self.assertTrue(torch.allclose(distribution.probs,ref,atol=1e-7))
        self.assertEqual(float(distribution.probs[:,2:].sum()),0)
        self.assertTrue(torch.allclose(reference_kl(distribution,ref),torch.zeros(2),atol=1e-6))

    def test_gae_stops_at_terminal(self):
        advantage,returns = gae(torch.tensor([1.,2.]),torch.tensor([0.5,0.25]),gamma=1,lam=1)
        torch.testing.assert_close(advantage,torch.tensor([2.5,1.75]))
        torch.testing.assert_close(returns,torch.tensor([3.,2.]))

    def test_actual_ppo_update_increases_rewarded_action_probability(self):
        torch.manual_seed(3)
        policy = ResidualPolicy()
        optimizer = torch.optim.Adam(policy.parameters(),lr=0.005)
        ref = torch.zeros(33)
        ref[:2] = 0.5
        obs = torch.zeros(64)
        episodes = []
        with torch.no_grad():
            distribution,value = policy.distribution(obs[None],ref[None])
            for index in range(128):
                action = index%2
                episodes.append([dict(observation=obs.clone(),reference=ref.clone(),action=action,
                                      value=value.item(),logp=distribution.log_prob(torch.tensor([action])).item(),
                                      reward=1.0 if action==1 else -1.0)])
        stats = ppo_update(policy,optimizer,episodes,torch.Generator().manual_seed(5),epochs=3)
        after,_ = policy.distribution(obs[None],ref[None])
        self.assertGreater(after.probs[0,1].item(),0.5)
        self.assertEqual(stats['decisions'],128)

    def test_features_do_not_include_future_pitch_or_key(self):
        observation = features(torch.ones(33)/33,72,48,2,480,7680,True,[48,52])
        self.assertEqual(observation.shape,(64,))
        self.assertTrue(torch.isfinite(observation).all())


class RewardTests(unittest.TestCase):
    def test_silence_and_flooding_do_not_beat_simple_support(self):
        lead = ['note',0,420,0,72,90]
        support = [['note',0,420,1,48,64],['note',0,420,1,52,64]]
        repeated = [['note',0,420,1,48,64]]*16
        flood = [['note',0,420,1,p,64] for p in range(36,67)]
        score = lambda notes: window_reward(lead,notes,0,480)
        self.assertGreater(score(support),score([]))
        self.assertGreater(score(support),score(repeated))
        self.assertGreater(score(support),score(flood))

    def test_fresh_corpus_deterministic_and_valid(self):
        for seed in [1000,2000,3000]:
            a = synthetic_episode(seed)
            self.assertEqual(a,synthetic_episode(seed))
            self.assertEqual(len(a[0]),16)
            self.assertTrue(all(n[1]%30==0 and n[2]%30==0 for n in a[0]))


if __name__=='__main__':
    unittest.main()
