"""Behavioral checks for the evaluation cases and scoring edge cases."""
import unittest

from evaluation_cases import build_cases, CHANGE, END
from music_metrics import diagnostics, adversarial_examples


class SuiteTests(unittest.TestCase):
    def test_interventions_share_the_prefix_and_do_not_leak_between_splits(self):
        cases = build_cases()
        self.assertEqual(len(cases),10)
        development_seeds, reserved_seeds = set(),set()
        for case in cases:
            target = development_seeds if case.split == "development" else reserved_seeds
            target.update(case.seeds)
            baseline = next(c for c in cases if c.family == case.family and c.condition == "steady")
            self.assertEqual([n for n in case.melody if n[1]<CHANGE],
                             [n for n in baseline.melody if n[1]<CHANGE])
            self.assertEqual(len(case.melody),32)
            self.assertTrue(all(n[1]%30==0 and n[2]%30==0 for n in case.melody))
            self.assertTrue(all(0 <= n[1] < n[1]+n[2] <= END for n in case.melody))
            self.assertEqual(case.melody, sorted(case.melody,key=lambda n:n[1]))
        self.assertFalse(development_seeds & reserved_seeds)

    def test_rhythm_change_preserves_pitches_but_changes_onsets(self):
        cases = build_cases()
        for family in ["theme_a","theme_b"]:
            steady = next(c for c in cases if c.family==family and c.condition=="steady")
            rhythm = next(c for c in cases if c.family==family and c.condition=="rhythm_change")
            self.assertEqual([n[4] for n in steady.melody],[n[4] for n in rhythm.melody])
            self.assertNotEqual([n[1] for n in steady.melody],[n[1] for n in rhythm.melody])


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.lead = [["note",0,480,0,72,90]]
        self.harmony = [dict(start=0,end=960,scale=[0,2,4,5,7,9,11],chord=[0,4,7])]

    def score(self,notes,start=0,end=960):
        return diagnostics(self.lead,notes,self.harmony,start,end)

    def test_silence_is_not_perfect_harmony(self):
        result = self.score([])
        self.assertIsNone(result["scale_fit_duration"])
        self.assertIsNone(result["intended_chord_fit_duration"])
        self.assertIsNone(result["rough_melody_interval_fraction"])
        self.assertEqual(result["sounding_coverage"],0)
        self.assertEqual(result["note_count"],0)

    def test_duration_weighting_and_coverage_are_different(self):
        notes = [["note",0,480,1,48,64],["note",0,240,1,49,64]]
        result = self.score(notes)
        self.assertAlmostEqual(result["scale_fit_duration"],2/3,places=5)
        self.assertEqual(result["sounding_coverage"],0.5)
        self.assertEqual(result["max_polyphony"],2)

    def test_notes_sustained_into_window_are_scored_without_counting_new_attacks(self):
        result = self.score([["note",0,720,1,48,64]],start=480)
        self.assertEqual(result["scale_fit_duration"],1)
        self.assertEqual(result["sounding_coverage"],0.5)
        self.assertEqual(result["note_count"],0)

    def test_known_rough_interval(self):
        self.lead = [["note",0,480,0,73,90]]
        result = self.score([["note",0,480,1,48,64]])
        self.assertEqual(result["rough_melody_interval_fraction"],1)

    def test_repetition_and_flooding_are_visible_even_if_tonal_score_is_good(self):
        case = build_cases()[0]
        fixtures = {name:diagnostics(case.melody,notes,case.harmony,0,END)
                    for name,notes in adversarial_examples().items()}
        self.assertEqual(fixtures["repeated_C"]["scale_fit_duration"],1)
        self.assertEqual(fixtures["repeated_C"]["dominant_pitch_share"],1)
        self.assertEqual(fixtures["repeated_C"]["pitch_entropy_bits"],0)
        self.assertEqual(fixtures["ascending_scale"]["scale_fit_duration"],1)
        self.assertEqual(fixtures["dense_chromatic"]["max_polyphony"],31)
        self.assertGreater(fixtures["dense_chromatic"]["notes_per_beat"],100)

    def test_reversed_window_is_rejected(self):
        with self.assertRaises(ValueError):
            self.score([],start=960,end=0)


if __name__ == "__main__":
    unittest.main()
