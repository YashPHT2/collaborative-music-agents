import unittest
import continue_note_trio as ct


class ContinuationTests(unittest.TestCase):
    def test_retain_starting_checkpoint_when_newer_is_worse(self):
        self.assertFalse(ct.should_select(dict(score=.5,eligible=True),dict(score=.7,eligible=True)))
        self.assertFalse(ct.should_select(dict(score=.7,eligible=True),dict(score=.7,eligible=True)))
        self.assertTrue(ct.should_select(dict(score=.71,eligible=True),dict(score=.7,eligible=True)))

    def test_collapsed_policy_cannot_win_selection(self):
        self.assertFalse(ct.should_select(dict(score=.95,eligible=False),dict(score=.5,eligible=True)))
        self.assertTrue(ct.should_select(dict(score=.4,eligible=True),dict(score=.9,eligible=False)))
        notes=[['note',i*480,480,ch,64,60] for i in range(16) for ch in [0,1,2]]
        ep=dict(notes=notes,brief=dict(bars=4),steps=[dict(reward=.9,components=dict(test=1))])
        self.assertFalse(ct.metrics(ep)['eligible'])

    def test_final_listening_seed_is_not_a_selection_seed(self):
        self.assertNotIn(ct.FINAL_SEED,ct.VALIDATION_SEEDS)
        self.assertNotEqual(ct.FINAL_SEED,9301001)


if __name__=='__main__': unittest.main()
