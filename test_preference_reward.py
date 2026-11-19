import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch

from preference_reward import canonical_waveform, window_starts, require_validated_reward
from ensemble_coherence_reward import sequence, disrupt, CoherenceRanker, pairwise_loss


class PreferenceRewardTests(unittest.TestCase):
    def test_cover_all_audio_without_random_crop(self):
        for length in (1, 480000, 576000, 1440001):
            starts = window_starts(length)
            coverage = np.zeros(length, dtype=bool)
            for start in starts:
                coverage[start:start + 480000] = True
            self.assertTrue(coverage.all())
            self.assertEqual(starts, window_starts(length))

    def test_mastering_gain_normalization(self):
        t = np.arange(48000) / 48000
        x = .04 * np.sin(2 * np.pi * 440 * t)
        a, _, _ = canonical_waveform(x, 48000)
        b, _, _ = canonical_waveform(x * .501187, 48000)
        np.testing.assert_allclose(a, b, atol=1e-7)

    def test_silence_and_invalid_input(self):
        y, rms, silent = canonical_waveform(np.zeros((480, 2)), 48000)
        self.assertTrue(silent)
        self.assertEqual(rms, 0)
        self.assertTrue(np.isfinite(y).all())
        for bad in ([], [np.nan], [np.inf]):
            with self.assertRaises(ValueError):
                canonical_waveform(bad, 48000)

    def test_sustain_rest_missing_are_distinct(self):
        x = sequence([['note', 0, 1920, 0, 60, 64], ['note', 2400, 480, 1, 67, 64]], 2).reshape(-1, 3, 19)
        self.assertEqual(x[1, 0, 12], 0)  # no new attack
        self.assertEqual(x[1, 0, 17], 1)  # still sounding
        self.assertEqual(x[17, 0, 17], 0)  # real rest
        self.assertEqual(x[17, 0, 18], 1)  # known part
        self.assertEqual(x[:, 2, 18].sum(), 0)  # missing part

    def test_one_role_changes_and_note_count_preserved(self):
        notes = [['note', i * 480, 240, r, 48 + r * 12 + i % 4, 64] for r in range(3) for i in range(8)]
        for kind in ('detune', 'desync', 'scramble'):
            result = disrupt(notes, 1, kind, 3840, 123)
            self.assertEqual(len(result), len(notes))
            self.assertEqual(sorted(n for n in notes if n[3] != 1), sorted(n for n in result if n[3] != 1))
            self.assertEqual(result, disrupt(notes, 1, kind, 3840, 123))
            self.assertNotEqual(sorted(notes), sorted(result))

    def test_scramble_retains_pitch_support_and_rhythm(self):
        notes = [['note', i * 480, 240, 0, 60 + i % 5, 64] for i in range(8)]
        result = disrupt(notes, 0, 'scramble', 3840, 42)
        self.assertEqual({n[4] for n in notes}, {n[4] for n in result})
        self.assertEqual([(n[1], n[2], n[5]) for n in notes], [(n[1], n[2], n[5]) for n in result])

    def test_no_direct_loss_for_missing_role_heads(self):
        torch.manual_seed(12)
        model = CoherenceRanker()
        loss = pairwise_loss(model, torch.randn(2, 32, 57), torch.randn(2, 32, 57), torch.tensor([0, 0]))
        loss.backward()
        self.assertGreater(float(model.head.weight.grad[0].abs().sum()), 0)
        self.assertEqual(float(model.head.weight.grad[1:].abs().sum()), 0)

    def test_synthetic_success_cannot_enable_rl(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'acceptance.json').write_text(json.dumps(dict(automated_gate_passed=True, rl_training_allowed=True)))
            with self.assertRaises(RuntimeError):
                require_validated_reward(path)


if __name__ == '__main__':
    unittest.main()
