import copy
import tempfile
import unittest
from pathlib import Path
import torch
import wave
import numpy as np
import mido
from unittest.mock import patch
import note_trio as nt
from band import ROLES as LEGACY_ROLES


class NoteTrioTests(unittest.TestCase):
    def test_variable_tempo_preserves_notes_and_duration(self):
        notes=[['note',0,480,0,60,64],['note',1920,1920,2,48,64]]
        tempo=[(0,60),(1920,120)]
        self.assertEqual(nt.tempo_seconds(2,100,tempo),6.)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'tempo.mid';nt.midi_file(path,notes,tempo_map=tempo)
            self.assertAlmostEqual(mido.MidiFile(path).length,6.)
            messages=[m for m in mido.MidiFile(path).tracks[0] if m.type=='set_tempo']
            self.assertEqual([m.time for m in messages],[0,1920])
        with self.assertRaises(ValueError):nt.tempo_seconds(2,100,[(0,100),(0,120)])

    def test_role_configuration_is_separate_from_legacy_band(self):
        self.assertEqual(set(nt.ROLES),{'organ','strings','cello'})
        self.assertEqual(set(LEGACY_ROLES),{'piano','bass','drums'})
        self.assertEqual(nt.ROLES['cello']['patch'],42)

    def test_twelve_pitch_classes_and_partner_removal(self):
        brief=nt.Brief()
        for pc in range(12):
            history=[['note',0,480,1,60+pc,60]]
            state=nt.state_features(brief,1,history)
            self.assertEqual(state[:12].argmax().item(),pc)
            self.assertEqual(nt.visible(history,1,'organ'),[])
        with self.assertRaises(ValueError): nt.state_features(brief,0,history)

    def test_atomic_simultaneous_commit_and_note_integrity(self):
        actions={'organ':[['note',0,240,0,64,60]],'strings':[['note',30,960,1,71,52]],'cello':[['note',0,480,2,40,58]]}
        a=nt.commit([],actions,0,nt.Brief())
        self.assertEqual(a,nt.commit([],dict(reversed(list(actions.items()))),0,nt.Brief()))
        broken=copy.deepcopy(actions); broken['cello'][0][4]=100
        ledger=[]
        with self.assertRaises(ValueError): nt.commit(ledger,broken,0,nt.Brief())
        self.assertEqual(ledger,[])
        with tempfile.TemporaryDirectory() as d:
            for role in [None,*nt.ROLES]: nt.midi_file(Path(d)/f'{role}.mid',a,role)

    def test_sustained_notes_and_no_future_reward(self):
        ledger=[['note',0,1440,2,40,60]]
        with self.assertRaises(ValueError):
            nt.commit(ledger,{'organ':[],'strings':[],'cello':[['note',480,480,2,43,60]]},1,nt.Brief())
        self.assertEqual(nt.reward([],1920),nt.reward([['note',1920,480,0,70,60]],1920))
        self.assertLess(nt.reward([],1920)[0],0)

    def test_audio_export_boundary_with_real_mastering(self):
        def synth_stub(path,wav):
            frames=(np.sin(np.arange(44100)*2*np.pi*440/44100)*3000).astype('<i2')
            with wave.open(str(wav),'wb') as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(44100); w.writeframes(frames.tobytes())
        ep=dict(notes=[['note',0,480,0,64,60]],decisions=[dict(note=['note',0,480,0,64,60])],
                trace=[],brief=dict(bars=1,bpm=100,tonic=4,minor=True))
        with tempfile.TemporaryDirectory() as d,patch.object(nt,'render',synth_stub):
            checks=nt.export(Path(d),'smoke',ep,stems=False)
            self.assertEqual(checks['mix']['seconds'],4.8)
            self.assertTrue((Path(d)/'smoke_mix.mp3').exists())


if __name__=='__main__': unittest.main()
