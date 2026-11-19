"""Checks for the accompaniment scheduler's timing and conditioning contract."""
import unittest

from accompany import accompany, melody, resolve_overlaps, CHANGE_TICK


class ScriptedSampler:
    def __init__(self, candidates):
        self.candidates = iter(candidates)
        self.committed = []
        self.context_at_sample = []

    def append_note(self, note, track):
        self.committed.append((track, note.copy()))

    def sample(self):
        self.context_at_sample.append([(track, note.copy()) for track, note in self.committed])
        return next(self.candidates)


class AccompanimentTests(unittest.TestCase):
    def test_future_melody_is_not_exposed_and_crossing_candidate_is_not_committed(self):
        lead = [["note", 0, 420, 0, 72, 90], ["note", 480, 420, 0, 78, 90]]
        first = ["note", 0, 120, 1, 48, 65]
        crossing = ["note", 600, 120, 1, 54, 65]
        second = ["note", 480, 120, 1, 50, 65]
        sampler = ScriptedSampler([first, crossing, second, None])
        result, stats = accompany(sampler, lead, end_tick=960)
        self.assertEqual(result, [first, second])
        self.assertEqual(stats["boundary_waits"], 1)
        self.assertEqual(stats["eos_waits"], 1)
        self.assertEqual([n[4] for t, n in sampler.context_at_sample[1] if t == 1], [72])
        self.assertEqual([n[4] for t, n in sampler.context_at_sample[2] if t == 1], [72, 78])
        self.assertNotIn((2, crossing), sampler.committed)

    def test_intervention_changes_only_the_second_half(self):
        original, changed = melody(), melody(True)
        self.assertEqual(original[:16], changed[:16])
        self.assertEqual(changed[16][1], CHANGE_TICK)
        for before, after in zip(original[16:], changed[16:]):
            self.assertEqual(after[4], before[4] + 2)
            self.assertEqual(before[:4], after[:4])
            self.assertEqual(before[5:], after[5:])

    def test_overlap_repair_preserves_other_pitches_and_input(self):
        notes = [["note", 0, 600, 1, 48, 60], ["note", 240, 480, 1, 52, 60],
                 ["note", 480, 240, 1, 48, 60]]
        cleaned = resolve_overlaps(notes)
        self.assertEqual(cleaned[0][2], 480)
        self.assertEqual(cleaned[1][2], 480)
        self.assertEqual(notes[0][2], 600)


if __name__ == "__main__":
    unittest.main()
