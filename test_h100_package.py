import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch
from h100_train import microchunks,rng_state,restore_rng,no_role_regression
from orchestra_learning import Sampler,ROLES


class PortableTrainerTests(unittest.TestCase):
    def test_chunk_limits_preserve_all_examples(self):
        items=[({'x':torch.zeros(n,8)},str(i)) for i,n in enumerate([4,5,8,9,10])]
        chunks=list(microchunks(items,3,20))
        self.assertEqual(sorted(r[1] for chunk in chunks for r in chunk),list('01234'))
        self.assertTrue(all(len(c)<=3 and len(c)*max(len(r[0]['x']) for r in c)<=20 for c in chunks))
        with self.assertRaises(ValueError):list(microchunks([({'x':torch.zeros(21,8)},)],3,20))

    def test_end_of_score_never_offers_empty_duration(self):
        s=object.__new__(Sampler);s.brief=SimpleNamespace(bars=6);s.beat=23
        s.settings={'grid':list(range(16)),'polyphony':3};s.observed=[];s.last_tick=0
        for onset in s.onsets():self.assertTrue(s.constrain_values('duration',[],{'time2':onset}))
        self.assertNotIn(15,s.onsets())

    def test_rng_restore_replays_draws(self):
        import random,numpy as np
        rng=rng_state();expected=(torch.rand(3),random.random(),np.random.rand())
        restore_rng(rng);actual=(torch.rand(3),random.random(),np.random.rand())
        torch.testing.assert_close(expected[0],actual[0]);self.assertEqual(expected[1:],actual[1:])

    def test_phrase_guard_cannot_hide_one_roles_regression(self):
        baseline={r:dict(event=1.,attributes=1.) for r in ROLES}
        improved={r:dict(event=.8,attributes=.8) for r in ROLES}
        self.assertTrue(no_role_regression(improved,baseline))
        improved['cellos']['attributes']=1.021
        self.assertFalse(no_role_regression(improved,baseline))
        improved['cellos']['attributes']=None
        self.assertFalse(no_role_regression(improved,baseline))


if __name__=='__main__':unittest.main()
