import json
from pathlib import Path
import tempfile
import unittest

import audit_supported_phrase_gate as g


def raw(start,duration,pitch=60):
    return dict(start=start,duration=duration,pitch=pitch,velocity=64,program=19)


class SupportedPhraseGateTests(unittest.TestCase):
    def test_boundary_sustain_keeps_original_attack_in_prefix(self):
        projected=[['note',7200,960,0,60,64]]
        notes,loss_start=g.slice_supported(projected,7680)
        self.assertEqual(loss_start,3840)
        self.assertEqual(notes,[['note',3360,960,0,60,64]])
        self.assertLess(notes[0][1],loss_start)
        self.assertGreater(notes[0][1]+notes[0][2],loss_start)
        g.validate_container(notes,loss_start)

    def test_prefix_loss_boundary_does_not_retrigger_note(self):
        projected=[['note',7200,960,0,60,64],['note',8000//30*30,480,0,64,64]]
        notes,loss_start=g.slice_supported(projected,7680)
        targets=[n for n in notes if n[1]>=loss_start]
        self.assertEqual(len(targets),1)
        self.assertEqual(targets[0][4],64)
        self.assertFalse(any(n[1]==loss_start and n[4]==60 for n in notes))

    def test_maximum_legal_carry_is_within_eight_beat_prefix(self):
        target=7680
        legal=[['note',target-g.CAP,g.CAP,0,60,64],['note',target-g.CAP+30,g.CAP,0,62,64]]
        notes,loss_start=g.slice_supported(legal,target)
        self.assertEqual(notes[1][1],30)
        self.assertEqual(notes[1][1]+notes[1][2]-loss_start,30)

    def test_sustaining_sparse_resting_missing_states(self):
        self.assertEqual(g.role_state([raw(7200,960)],[['note',7200,960,0,60,64]],7680,'organ'),('present_sustaining',True))
        self.assertEqual(g.role_state([raw(8000,960)],[['note',7980,960,0,60,64]],7680,'organ'),('present_sparse',True))
        self.assertEqual(g.role_state([raw(7000,100)],[],7680,'organ'),('present_resting',True))
        self.assertEqual(g.role_state([],[],7680,'organ'),('missing',False))

    def test_pruning_activity_is_not_labelled_genuine_rest(self):
        self.assertEqual(g.role_state([raw(8000,960)],[],7680,'organ'),('missing',False))

    def test_duration_boundary_is_strict(self):
        self.assertFalse(raw(0,3840)['duration']>g.CAP)
        self.assertTrue(raw(0,3841)['duration']>g.CAP)
        self.assertTrue(g.overlap(raw(1920,22*480),7680,15360))

    def test_group_gate_counts_unique_compositions_not_windows(self):
        rows=[dict(split='validation',group='same-work',eligible=True,roles={r:dict(gate_passed=True) for r in g.ROLES}) for _ in range(50)]
        result=g.group_gate(rows,10)
        self.assertFalse(result['passed'])
        self.assertEqual(result['validation_groups_per_role'],{r:1 for r in g.ROLES})

    def test_ten_and_twelve_requirements_remain_distinct(self):
        rows=[dict(split='validation',group=str(i),eligible=True,roles={r:dict(gate_passed=True) for r in g.ROLES}) for i in range(10)]
        self.assertTrue(g.group_gate(rows,10)['passed'])
        self.assertFalse(g.group_gate(rows,12)['passed'])

    def test_gate_refuses_count_pass_without_verified_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'summary.json'
            path.write_text(json.dumps(dict(gates={'train_offline_ensemble':dict(passed=True)},training_loader_integrated=False)))
            with self.assertRaisesRegex(RuntimeError,'loader'):
                g.require_supervised_gate(directory)


if __name__=='__main__':unittest.main()
