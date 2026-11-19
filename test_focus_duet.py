import tempfile
from pathlib import Path
import unittest
import numpy as np
import note_trio as nt
from generate_focus_duet import duet_roles,DUET,layer_weights,midi


class FocusDuetTests(unittest.TestCase):
    def test_role_context_restored_on_failure(self):
        original=nt.ROLES
        with self.assertRaises(RuntimeError):
            with duet_roles():
                self.assertEqual(set(nt.ROLES),{'piano','cello'})
                raise RuntimeError('probe')
        self.assertIs(nt.ROLES,original)

    def test_velocity_crossfade_has_no_gain_hole(self):
        weights=np.array([layer_weights(v) for v in range(1,54)])
        np.testing.assert_allclose((weights*weights).sum(axis=1),1,atol=1e-12)
        self.assertTrue((weights>=0).all())
        self.assertEqual(layer_weights(20),(1.,0.))
        self.assertEqual(layer_weights(48),(0.,1.))

    def test_midi_accepts_shared_onsets_and_note_releases(self):
        with tempfile.TemporaryDirectory() as temporary:
            notes=[['note',0,480,0,60,32],['note',0,960,1,48,36],['note',480,480,0,60,40]]
            midi(Path(temporary)/'duet.mid',notes)
        self.assertEqual(DUET['piano']['patch'],0)


if __name__=='__main__':unittest.main()
