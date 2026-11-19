import io
import itertools
import unittest
import mido

from audit_functional_roles import parse_with_tracks, interval_coverage, statistics, qualify, matching


def note(start=0, duration=7680, pitch=60, program=19, track=1, channel=0):
    return dict(start=start, duration=duration, pitch=pitch, velocity=64, program=program,
                source_track=track, source_channel=channel, source_name='test')


def candidate(role, track, pitch, provenance='transfer'):
    return dict(source_id=f'{track}:{role}', source_track=track, provenance=provenance,
                stats=dict(median_pitch=pitch), weight=2 if provenance == 'exact' else 1)


class FunctionalRoleAuditTests(unittest.TestCase):
    def test_sustained_note_crossing_window_is_sounding_without_new_attack(self):
        notes = [note(start=-480, duration=9000)]
        stats = statistics(notes, 0, 7680)
        self.assertEqual(stats['duty'], 1)
        self.assertEqual(stats['onset_groups'], 0)
        self.assertEqual(stats['mean_duration_beats'], 16)

    def test_coverage_counts_overlap_once_and_distinguishes_polyphony(self):
        duty, poly, maximum = interval_coverage([note(duration=3840), note(start=1920, duration=3840, pitch=64)], 0, 7680)
        self.assertEqual(duty, .75)
        self.assertAlmostEqual(poly, 1/3)
        self.assertEqual(maximum, 2)

    def test_sustained_exact_organ_does_not_need_more_notes(self):
        provenance, rejected, _ = qualify('organ', [note()], 0, 7680)
        self.assertEqual(provenance, 'exact')
        self.assertEqual(rejected, [])

    def test_pad_chord_qualifies_and_staccato_pad_rejected(self):
        notes = [note(pitch=pitch, program=89) for pitch in (48, 55, 60)]
        provenance, rejected, _ = qualify('organ', notes, 0, 7680)
        self.assertEqual(provenance, 'transfer')
        self.assertEqual(rejected, [])
        provenance, rejected, _ = qualify('organ', [note(start=i*240, duration=60, program=89) for i in range(32)], 0, 7680)
        self.assertIsNone(provenance)
        self.assertIn('mean_duration_below_1_beat', rejected)
        self.assertIn('onset_density_above_6_groups_per_bar', rejected)

    def test_range_filter_cannot_disguise_high_bassoon_as_cello(self):
        notes = [note(start=i*480, duration=480, pitch=80, program=70) for i in range(12)] + [note(pitch=48, program=70)]
        provenance, rejected, _ = qualify('cello', notes, 0, 7680)
        self.assertIsNone(provenance)
        self.assertIn('target_range_fraction_below_0.80', rejected)
        self.assertIn('median_outside_36_62', rejected)

    def test_assignment_never_duplicates_physical_track(self):
        options = {'organ': [candidate('organ', 1, 60)],
                   'strings': [candidate('strings', 1, 70), candidate('strings', 2, 70)],
                   'cello': [candidate('cello', 3, 48)]}
        result, _ = matching(options)
        self.assertEqual([result[r]['source_track'] for r in ('organ', 'strings', 'cello')], [1, 2, 3])

    def test_register_constraint_uses_joint_assignment(self):
        options = {'organ': [], 'strings': [candidate('strings', 1, 58)],
                   'cello': [candidate('cello', 2, 60), candidate('cello', 3, 48)]}
        result, reasons = matching(options)
        self.assertEqual(result['cello']['source_track'], 3)
        self.assertGreater(reasons['incompatible_register_combinations'], 0)

    def test_exact_source_crossing_is_not_relabelled_as_ineligible(self):
        result, _ = matching({'organ': [], 'strings': [candidate('strings', 1, 60, 'exact')],
                              'cello': [candidate('cello', 2, 62, 'exact')]})
        self.assertTrue(result['strings'] and result['cello'])

    def test_matching_is_global_optimum_with_dummy_missing_roles(self):
        options = {'organ': [candidate('organ', 1, 60, 'exact'), candidate('organ', 2, 60)],
                   'strings': [candidate('strings', 1, 70, 'exact'), candidate('strings', 3, 68)],
                   'cello': [candidate('cello', 2, 48, 'exact'), candidate('cello', 3, 72)]}
        result, _ = matching(options)
        maximum = -1
        for combination in itertools.product(*[[*options[r], None] for r in ('organ', 'strings', 'cello')]):
            parts = [c for c in combination if c]
            if len({c['source_track'] for c in parts}) != len(parts): continue
            strings, cello = combination[1:]
            if strings and cello and 'transfer' in (strings['provenance'], cello['provenance']) and strings['stats']['median_pitch'] <= cello['stats']['median_pitch']: continue
            maximum = max(maximum, sum(c['weight'] for c in parts))
        self.assertEqual(sum(c['weight'] for c in result.values() if c), maximum)

    def test_recover_track_channel_and_global_program_with_no_new_notes(self):
        midi = mido.MidiFile(ticks_per_beat=480)
        meta = mido.MidiTrack(); midi.tracks.append(meta)
        meta.append(mido.Message('program_change', channel=1, program=42, time=0))
        track = mido.MidiTrack(); midi.tracks.append(track)
        track.append(mido.MetaMessage('track_name', name='Cello source'))
        track.append(mido.Message('note_on', channel=1, note=48, velocity=70, time=120))
        track.append(mido.Message('note_off', channel=1, note=48, velocity=0, time=960))
        blob = io.BytesIO(); midi.save(file=blob)
        parsed = parse_with_tracks(blob.getvalue())
        self.assertEqual(len(parsed['notes']), 1)
        result = parsed['notes'][0]
        self.assertEqual((result['source_track'], result['source_channel'], result['program']), (1, 1, 42))
        self.assertEqual((result['start'], result['duration'], result['pitch'], result['velocity']), (120, 960, 48, 70))
        self.assertEqual(result['source_name'], 'Cello source')


if __name__ == '__main__':
    unittest.main()
