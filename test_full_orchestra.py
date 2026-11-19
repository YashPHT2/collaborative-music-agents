import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import mido
import generate_full_orchestra as orchestra


class OrchestraTests(unittest.TestCase):
    def test_private_part_commit_and_same_pitch_in_distinct_pianos(self):
        actions={r:[] for r in orchestra.ROLES}
        for r in ('piano_anchor','piano_motion'):
            actions[r]=[['note',0,480,orchestra.ROLES[r]['id'],60,40]]
        result=orchestra.commit([],actions,0,12)
        self.assertEqual(len(result),2)
        with self.assertRaises(ValueError):orchestra.commit([],{'piano_anchor':[]},0,12)

    def test_shared_model_channel_does_not_alias_history_ownership(self):
        sampler=object.__new__(orchestra.OrchestraSampler)
        sampler.role_id=orchestra.ROLES['piano_motion']['id']
        note=['note',0,480,orchestra.CHANNELS[0],60,40]
        self.assertFalse(sampler.own(note+[orchestra.ROLES['piano_anchor']['id']]))
        self.assertTrue(sampler.own(note+[sampler.role_id]))
        self.assertTrue(sampler.own(note))

    def test_nineteen_part_midi_preserves_unique_ports_and_notes(self):
        notes=[['note',0,480,s['id'],s['pitches'][0],40] for s in orchestra.ROLES.values()]
        with tempfile.TemporaryDirectory() as temporary,patch.object(orchestra,'OUT',Path(temporary)):
            orchestra.export_midis(dict(notes=notes))
            midi=mido.MidiFile(Path(temporary)/'orchestra_multiport.mid');addresses=[]
            for track in midi.tracks[1:]:
                port=next(m.port for m in track if m.type=='midi_port')
                channel=next(m.channel for m in track if m.type=='program_change')
                addresses.append((port,channel))
            self.assertEqual(len(addresses),19)
            self.assertEqual(len(set(addresses)),19)
            self.assertTrue(all(channel!=9 for port,channel in addresses))


if __name__=='__main__':unittest.main()
