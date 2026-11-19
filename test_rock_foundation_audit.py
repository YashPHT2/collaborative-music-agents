import unittest
import audit_rock_foundation as a

class Tests(unittest.TestCase):
    def test_crop_preserves_notes_without_fake_attacks(self):
        notes=[['note',0,480,0,60,80],['note',3840,960,2,40,84],['note',11520,60,2,42,80]]
        self.assertEqual(a.crop(notes),[['note',0,960,2,40,84]])
        self.assertEqual(notes[1][1],3840)
    def test_crop_rejects_crossing_sustains(self):
        with self.assertRaises(ValueError):a.crop([['note',3780,120,0,60,80]])
    def test_out_of_domain_pitch_count(self):
        result=a.role_metrics([['note',0,120,2,40,80],['note',480,120,2,82,80]],'bass',{'min':25,'max':57})
        self.assertEqual(result['outside_training_pitch_support'],1)
        self.assertEqual(result['pitch_max'],82)

if __name__=='__main__':unittest.main()
