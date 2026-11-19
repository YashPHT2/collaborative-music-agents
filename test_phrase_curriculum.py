import unittest
import torch
import phrase_curriculum as pc


class PhraseTests(unittest.TestCase):
    def test_melody_analysis_prefers_motion_and_motif_over_drone(self):
        notes=[['note',i*240,120,0,p,64] for i,p in enumerate([60,62,64,65,67,65,64,62]*2)]
        w=dict(notes=notes)
        self.assertTrue(pc.preferred(w,'organ'))
        self.assertGreater(pc.melodic_features(notes,0)['repeated_motifs'],0)
        self.assertFalse(pc.preferred(dict(notes=[['note',i*240,120,0,60,64] for i in range(16)]),'organ'))

    def test_contour_is_transposition_invariant_and_has_gradients(self):
        prediction=torch.tensor([60.,62.,64.,65.],requires_grad=True)
        target=torch.tensor([67.,69.,71.,72.])
        good=pc.contour_loss([(prediction,target)],[0])
        self.assertEqual(float(good),0.)
        bad=torch.tensor([60.,58.,56.,55.],requires_grad=True)
        loss=pc.contour_loss([(bad,target)],[0]);loss.backward()
        self.assertGreater(float(loss),0.)
        self.assertTrue(torch.isfinite(bad.grad).all())

    def test_contour_does_not_bridge_different_phrases(self):
        stats=[(torch.tensor([60.]),torch.tensor([60.])),(torch.tensor([80.]),torch.tensor([40.]))]
        self.assertIsNone(pc.contour_loss(stats,[0,1]))


if __name__=='__main__':unittest.main()
