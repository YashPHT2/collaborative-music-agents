import unittest
import copy
import orchestra_rl_selection as s

class SelectionTests(unittest.TestCase):
    def baseline(self):
        return [dict(seed=seed,reward=.1,roles={r:dict(sounding=.3) for r in s.l.ROLES}) for seed in s.VALIDATION_SEEDS]
    def test_rejects_role_disappearance_despite_team_gain(self):
        baseline=self.baseline();candidate=copy.deepcopy(baseline)
        for row in candidate:row['reward']=.3;row['roles']['cellos']['sounding']=0.
        self.assertFalse(s.select(candidate,baseline,.1)['passed'])
    def test_rejects_missing_seeds(self):
        with self.assertRaises(ValueError):s.select(self.baseline()[:-1],self.baseline(),.1)
    def test_accepts_paired_improvement_without_participation_regression(self):
        baseline=self.baseline();candidate=copy.deepcopy(baseline)
        for row in candidate:row['reward']=.3
        self.assertTrue(s.select(candidate,baseline,.1)['passed'])

if __name__=='__main__':unittest.main()
