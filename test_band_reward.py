import copy
import unittest
from band import Brief,BandEnv
from band_reward import ensemble_reward_v2

def fixture():
    result=[]
    for bar in range(4):
        for beat,p in enumerate([60,64,67,65+bar%2]):
            t=bar*1920+beat*480
            result.extend([['note',t,360,0,p,75],['note',t,480,1,36,65],['note',t,120,9,36 if beat%2==0 else 38,60]])
    return result

class RewardTests(unittest.TestCase):
    def setUp(self): self.b=Brief(4,reward_version='band-heuristic-v2'); self.notes=fixture()
    def score(self,ns,start=5760,end=7680): return ensemble_reward_v2(ns,start,end,self.b)
    def test_sanity_failures(self):
        good,_=self.score(self.notes)
        empty,_=self.score([])
        flood=self.notes+[['note',i*120,120,0,48+j,100] for i in range(64) for j in range(12)]
        self.assertGreater(good,empty); self.assertGreater(good,self.score(flood)[0])
    def test_transposition_and_input_order(self):
        edited=copy.deepcopy(self.notes)
        for n in edited:
            if n[3]!=9: n[4]+=2
        self.assertAlmostEqual(self.score(edited)[0],self.score(list(reversed(self.notes)))[0])
    def test_future_is_hidden(self):
        future=[['note',8000,480,0,84,90]]
        self.assertEqual(self.score(self.notes),self.score(self.notes+future))
    def test_arrangement_has_gradient_and_rewards_space(self):
        good,parts=self.score(self.notes)
        busy=self.notes+[['note',i*120,120,9,p,70] for i in range(48,64) for p in [42,46,49]]
        self.assertGreater(parts['arrangement'],self.score(busy)[1]['arrangement'])
        self.assertGreater(parts['arrangement'],0)
    def test_resolving_tension(self):
        base=[['note',0,1920,1,36,65],['note',0,480,0,61,75],['note',480,480,0,60,75]]
        stuck=copy.deepcopy(base); stuck[-1][4]=61
        self.assertGreater(self.score(base,0,1920)[1]['harmony'],self.score(stuck,0,1920)[1]['harmony'])
    def test_repeated_chord_voicing_does_not_create_melody_runs(self):
        a=[['note',0,480,0,72,70],['note',480,480,0,74,70],['note',960,480,0,76,70]]
        from band_reward import topline
        self.assertEqual(topline(a),topline(a+[['note',0,480,0,60,70],['note',480,480,0,62,70]]))
    def test_environment_selects_version(self):
        env=BandEnv(); env.reset(self.b,1)
        parts={}
        for _ in range(4): _,_,_,parts=env.step(dict(piano=[],bass=[],drums=[]))
        self.assertIn('participation',parts)

if __name__=='__main__': unittest.main()
