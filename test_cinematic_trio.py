import tempfile
import unittest
from pathlib import Path
import torch
import cinematic_trio as ct


class CinematicTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2); torch.manual_seed(7)
        self.agents={r:ct.PhraseAgent() for r in ct.ROLES}
        self.critic=torch.nn.Linear(ct.OBS,1)
        self.theme=[1,2,3,4,3,2,1,0]*2

    def test_simultaneous_order_and_causal_observations(self):
        a=ct.rollout(self.agents,self.critic,self.theme,15,bars=8)
        b=ct.rollout(self.agents,self.critic,self.theme,15,bars=8,order=list(reversed(ct.ROLES)))
        self.assertEqual(a['notes'],b['notes'])
        with self.assertRaises(ValueError): ct.observation(a['notes'],0,8)
        self.assertEqual(a['trace'][0]['history_hash'],b['trace'][0]['history_hash'])

    def test_all_roles_valid_and_exact_midi_roundtrip(self):
        ep=ct.rollout(self.agents,self.critic,self.theme,17,bars=24)
        self.assertEqual({n[3] for n in ep['notes']},{0,1,2})
        self.assertTrue(all(n[1]>=0 and 0<n[2] and n[1]+n[2]<=24*ct.BAR for n in ep['notes']))
        with tempfile.TemporaryDirectory() as d:
            for r in [None,*ct.ROLES]:
                p=Path(d)/f'{r}.mid'; ct.write_midi(p,ep['notes'],r)
                self.assertGreater(ct.verify_midi(p,ep['notes'],r),0)

    def test_actor_independence_and_joint_gradients(self):
        before={r:ct.fingerprint(a) for r,a in self.agents.items()}
        old=ct.OUT
        with tempfile.TemporaryDirectory() as d:
            try:
                ct.OUT=Path(d); ct.train(self.agents,self.critic,self.theme,updates=2)
            finally: ct.OUT=old
        self.assertTrue(all(ct.fingerprint(a)!=before[r] for r,a in self.agents.items()))
        params=[set(p.data_ptr() for p in a.parameters()) for a in self.agents.values()]
        self.assertFalse(params[0]&params[1] or params[0]&params[2] or params[1]&params[2])

    def test_all_twelve_pitch_classes_are_distinct(self):
        self.assertEqual(ct.observation([],0,24)[2:14].sum().item(),1)
        self.assertGreater(ct.energy(18,24),ct.energy(0,24))
        self.assertGreater(ct.energy(18,24),ct.energy(23,24))


if __name__=='__main__': unittest.main()
