import unittest
from evaluate_supported_ensemble import intervene,response_metrics,summarize


class ComparisonTests(unittest.TestCase):
    def test_intervention_changes_only_partner_observations(self):
        notes=[['note',3840,480,0,60,64],['note',3840,480,1,64,64]]
        self.assertEqual(intervene(notes,'organ','missing'),notes[:1])
        shifted=intervene(notes,'organ','mismatched')
        self.assertEqual(shifted[0],notes[0]);self.assertEqual(shifted[1][4],70)
        self.assertEqual(notes[1][4],64)

    def test_response_ignores_partner_partner_harmony(self):
        notes=[['note',3840,480,0,60,64],['note',3840,480,1,64,64],['note',3840,480,2,61,64]]
        result=response_metrics(notes,notes[:1],'organ')
        self.assertEqual(result['consonant_overlap_fraction'],.5)
        self.assertEqual(result['overlapping_note_pair_beats'],2)
        self.assertEqual(result['sounding_beats'],1)

    def test_silent_response_is_not_perfect_harmony(self):
        result=response_metrics([['note',3840,480,1,64,64]],[],'organ')
        self.assertIsNone(result['consonant_overlap_fraction'])
        self.assertEqual(result['sounding_beats'],0)
        self.assertEqual(summarize([])['organ']['compositions'],0)


if __name__=='__main__':unittest.main()
