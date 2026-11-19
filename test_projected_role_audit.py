import io
from pathlib import Path
import tempfile
import unittest
import mido

import audit_projected_roles as p


def raw(start, duration, pitch, program=19, source_id=0):
    return dict(start=start, duration=duration, pitch=pitch, velocity=64, program=program,
                source_note_id=source_id, source_track=1, source_channel=0, source_name='test')


class ProjectedRoleAuditTests(unittest.TestCase):
    def test_incoming_sustain_counted_raw_but_omitted_by_legacy_crop(self):
        parts = dict(organ=[raw(7200, 1000, 60)], strings=[], cello=[])
        before = p.raw_window(parts, 7680)
        self.assertGreater(p.metric(before['organ'])['duty'], 0)
        self.assertEqual(p.metric(before['organ'])['onset_groups'], 0)
        notes, _, changes = p.canonical_window(parts, 7680)
        self.assertEqual(notes, [])
        self.assertEqual(changes['organ']['incoming_sustain_omitted_by_onset_crop'], 1)

    def test_long_source_sustain_lost_to_cap_is_not_misattributed_to_crop(self):
        parts = dict(organ=[raw(0,10000,60)], strings=[], cello=[])
        notes, _, changes = p.canonical_window(parts,7680)
        self.assertEqual(notes,[])
        self.assertEqual(changes['organ']['incoming_sustain_lost_before_crop_by_duration_cap_or_cleanup'],1)
        self.assertEqual(changes['organ']['incoming_sustain_omitted_by_onset_crop'],0)

    def test_duration_cap_is_eight_beats(self):
        notes, _, _ = p.canonical_window(dict(organ=[raw(0,7680,60)], strings=[], cello=[]), 0)
        self.assertEqual(notes[0][2], 3840)

    def test_canonicalization_matches_existing_for_exact_source(self):
        import ensemble_data as ed
        parts = dict(organ=[raw(12,7890,60,source_id=0),raw(970,710,60,source_id=1),raw(1000,500,95,source_id=2)],strings=[],cello=[])
        actual, _, _ = p.canonical_window(parts,0)
        expected, _ = ed.canonicalize(parts['organ'],max_duration=3840)
        self.assertEqual(actual,expected['organ'])

    def test_projection_agrees_and_keeps_lowest_cello_voice(self):
        notes = [['note',0,960,2,48,64],['note',0,960,2,55,64],['note',480,960,2,50,64]]
        projected, changes = p.traced_projection(notes)
        self.assertEqual(projected,[['note',0,480,2,48,64],['note',480,960,2,50,64]])
        self.assertEqual(changes['cello']['simultaneous_cello_voice_discarded'],1)
        self.assertEqual(changes['cello']['cello_overlap_shortened'],1)

    def test_zero_polyphony_ratio_is_undefined_not_failure(self):
        self.assertIsNone(p.ratio(0,0))
        self.assertEqual(p.ratio(0,1),0)

    def test_fingerprint_preserves_roles_and_aligned_silence(self):
        notes = [['note',480,480,0,60,64],['note',480,480,2,48,64]]
        transposed = [['note',n[1],n[2],n[3],n[4]+2,n[5]] for n in notes]
        self.assertEqual(p.fingerprint(notes,True),p.fingerprint(transposed,True))
        shifted = [['note',n[1]+480,n[2],n[3],n[4],n[5]] for n in notes]
        self.assertNotEqual(p.fingerprint(notes,True),p.fingerprint(shifted,True))
        swapped = [['note',n[1],n[2],2-n[3],n[4],n[5]] for n in notes]
        self.assertNotEqual(p.fingerprint(notes,True),p.fingerprint(swapped,True))

    def test_cross_split_overlap_excludes_entire_lower_priority_group(self):
        def row(i, split, group, fp):
            return dict(id=i,split=split,group=group,passes=True,exact_fingerprint=fp,transposition_fingerprint=fp)
        rows = [row('a','test','heldout','shared'),row('b','train','work','shared'),row('c','train','work','unique'),row('d','train','other','safe')]
        kept, log = p.deduplicate(rows,'passes')
        self.assertEqual({r['id'] for r in kept},{'a','d'})
        self.assertEqual(len(log['excluded_composition_groups']),1)

    def test_raw_preview_retains_overlapping_identical_pitches(self):
        notes = [['note',0,960,0,60,64],['note',480,960,0,60,80],['note',0,960,2,48,64]]
        with tempfile.TemporaryDirectory() as directory:
            result = p.inspection_midi(Path(directory)/'raw.mid',notes)
        self.assertTrue(result['roundtrip_exact'])
        self.assertEqual(result['notes'],3)

    def test_anomaly_notes_are_reported_without_inventing_offsets(self):
        midi=mido.MidiFile();track=mido.MidiTrack();midi.tracks.append(track)
        track.append(mido.Message('note_off',note=65,velocity=0))
        track.append(mido.Message('note_on',note=60,velocity=64))
        track.append(mido.Message('note_off',note=60,velocity=0))
        track.append(mido.Message('note_on',note=62,velocity=64,time=480))
        blob=io.BytesIO();midi.save(file=blob)
        kinds={r['kind'] for r in p.source_event_anomalies(blob.getvalue())}
        self.assertEqual(kinds,{'unmatched_note_off','zero_or_negative_duration','unfinished_note_on'})


if __name__=='__main__':
    unittest.main()
