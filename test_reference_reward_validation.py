import unittest
import numpy as np
import validate_reference_reward as v


class ReferenceRewardTests(unittest.TestCase):
    def test_source_splits_have_no_overlap(self):
        for split in v.SPLITS.values():
            self.assertFalse(any(v.interval_overlap(a,b) for a in split['calibration'] for b in split['heldout']))

    def test_transforms_keep_length_and_are_deterministic(self):
        t=np.arange(v.SR*3)/v.SR
        x=np.stack([.2*np.sin(2*np.pi*220*t),.2*np.sin(2*np.pi*330*t)],axis=1).astype(np.float32)
        for name in ['phrase_shuffle','band_desync','short_loop','gain_minus_6db','mild_room','noise','silence']:
            a=v.perturb(x,name,1);b=v.perturb(x,name,1)
            self.assertEqual(a.shape,x.shape)
            self.assertTrue(np.array_equal(a,b))
            self.assertTrue(np.isfinite(a).all())

    def test_silence_is_finite_and_gets_zero_reward(self):
        f,rms=v.features(np.zeros((v.SR*12,2),dtype=np.float32))
        profile=v.calibrate([f,f,f])
        reward,parts=v.score(profile,f,rms)
        self.assertEqual(reward,0.)
        self.assertTrue(all(np.isfinite(a).all() for a in f.values()))

    def test_gain_invariance(self):
        rng=np.random.default_rng(4);x=rng.normal(0,.05,(v.SR*12,2)).astype(np.float32)
        a,ra=v.features(x);b,rb=v.features(x*.5)
        for name in a:self.assertTrue(np.allclose(a[name],b[name],atol=2e-4,rtol=1e-3),name)
        profile=v.calibrate([a,a,a]);sa,_=v.score(profile,a,ra);sb,_=v.score(profile,b,rb)
        self.assertLess(abs(sa-sb),1e-4)


if __name__=='__main__':unittest.main()
