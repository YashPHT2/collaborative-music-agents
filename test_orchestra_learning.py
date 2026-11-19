import copy
import unittest
import torch
from types import SimpleNamespace
import orchestra_learning as l


class LearningTests(unittest.TestCase):
    def test_last_onset_can_fit_minimum_duration(self):
        sampler=object.__new__(l.Sampler)
        sampler.beat=23;sampler.brief=SimpleNamespace(bars=6)
        sampler.settings=dict(grid=list(range(16)),polyphony=3)
        sampler.observed=[];sampler.last_tick=0
        self.assertIn(14,sampler.onsets())
        self.assertNotIn(15,sampler.onsets())
        for step in sampler.onsets():
            self.assertTrue(sampler.constrain_values('duration',[],{'time2':step}))

    def test_missing_and_prefix_have_no_teacher_loss(self):
        w=dict(available_roles=[],notes=[])
        self.assertIsNone(l.teacher(None,w,'pipe_organ',8))
        w['available_roles']=['pipe_organ']
        self.assertIsNone(l.teacher(None,w,'pipe_organ',7))

    def test_partner_announcements_reject_own_future_targets(self):
        with self.assertRaises(ValueError):l.draft_features([['note',3840,480,0,60,40]],'pipe_organ',8)
        valid=l.draft_features([['note',3840,480,l.ROLES['cellos']['id'],48,40]],'pipe_organ',8)
        self.assertEqual(valid.shape,(1,8))

    def test_sustain_is_not_silence(self):
        w=dict(notes=[['note',0,3840,0,60,40]])
        self.assertEqual(l.kind(w,'pipe_organ',1),'sustain')
        self.assertEqual(l.kind(w,'pipe_organ',8),'silence')

    def test_plan_prefix_is_causal(self):
        torch.manual_seed(31);planner=l.Planner().eval();goal=torch.tensor([.1,.3,.4,1.])
        prefix=[['note',0,960,0,60,40]];future=[['note',3840,960,0,84,64]]
        a=l.plan_sequence(planner,prefix,6,['pipe_organ'],goal)
        b=l.plan_sequence(planner,prefix+future,6,['pipe_organ'],goal)
        torch.testing.assert_close(a[:4],b[:4])


if __name__=='__main__':unittest.main()
