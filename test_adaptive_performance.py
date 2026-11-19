import copy,unittest
import numpy as np
from adaptive_performance import PhraseSampler,infer_intent
from band import Brief

class PhraseTests(unittest.TestCase):
    def sampler(self,role,beat=0):
        sampler=object.__new__(PhraseSampler)
        sampler.role=role;sampler.beat=beat;sampler.brief=Brief(12)
        sampler.intent=dict(chord=[9,0,4],root=9,activity=.5)
        return sampler
    def test_phrase_duration_cannot_cross_bar(self):
        for role in ['piano','bass','drums']:
            s=self.sampler(role,3)
            self.assertTrue(all(v<=8 for v in s.constrain_values('duration',list(range(4,65,4)),dict(time2=8))))
    def test_drum_velocity_is_restrained(self):
        values=self.sampler('drums').constrain_values('velocity',list(range(40,101,4)),{})
        self.assertLessEqual(max(values),56);self.assertGreaterEqual(min(values),28)
    def test_bass_follows_announced_harmony(self):
        values=self.sampler('bass').constrain_values('pitch',list(range(28,56)),{})
        self.assertTrue(values);self.assertTrue(all(p%12 in [9,0,4] for p in values))
    def test_transposed_leader_transposes_intent(self):
        lead=[['note',i*480,480,0,p,72] for i,p in enumerate([69,72,76,72])]
        shifted=copy.deepcopy(lead)
        for n in shifted:n[4]+=2
        old=infer_intent(lead,9); new=infer_intent(shifted,11)
        self.assertEqual((old['root']+2)%12,new['root'])
        self.assertEqual(sorted((p+2)%12 for p in old['chord']),sorted(new['chord']))

if __name__=='__main__':unittest.main()
