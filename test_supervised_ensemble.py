import copy
import io
import unittest
from types import SimpleNamespace

import mido
import torch
import note_trio as nt
import ensemble_data as ed
from midi_tokenizer import MIDITokenizer
from sequence_critic import SequenceCritic,critic_state,TOKENS,FEATURES
from supervised_ensemble import teacher_example,windows,playable_projection


class EnsembleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=SimpleNamespace(tokenizer=MIDITokenizer('v2'),device='cpu')

    def test_program_at_onset_and_drums_excluded(self):
        midi=mido.MidiFile(ticks_per_beat=480);track=mido.MidiTrack();midi.tracks.append(track)
        track.extend([mido.Message('program_change',channel=0,program=42),
                      mido.Message('note_on',channel=0,note=48,velocity=60),
                      mido.Message('program_change',channel=0,program=0,time=120),
                      mido.Message('note_on',channel=9,note=60,velocity=80),
                      mido.Message('note_off',channel=0,note=48,time=120),
                      mido.Message('note_off',channel=9,note=60)])
        out=io.BytesIO();midi.save(file=out)
        notes,_,_=ed.parse_midi(out.getvalue())
        self.assertEqual(len(notes),1);self.assertEqual(notes[0]['program'],42)
        roles,_=ed.canonicalize(notes)
        self.assertEqual(len(roles['cello']),1);self.assertFalse(roles['organ'])

    def test_no_proxy_and_duration_grid_after_overlap_trim(self):
        raw=[dict(start=s,duration=200,pitch=48,velocity=60,program=42) for s in [0,90]]
        raw.append(dict(start=0,duration=240,pitch=48,velocity=60,program=32))
        roles,_=ed.canonicalize(raw)
        self.assertEqual(len(roles['cello']),2)
        self.assertTrue(all(n[2]%60==0 for n in roles['cello']))
        self.assertFalse(roles['organ']);self.assertFalse(roles['strings'])

    def example(self):
        return dict(song='test',split='train',beat=4,roles={},notes=[
            ['note',0,480,0,64,60],['note',0,480,1,71,60],['note',0,480,2,40,60],
            ['note',1920,480,0,67,60],['note',1920,480,1,72,60],['note',1920,480,2,43,60]])

    def test_teacher_partner_future_hidden_and_all_labels_legal(self):
        example=self.example()
        for role in nt.ROLES:
            a=teacher_example(self.model,role,example)
            changed=copy.deepcopy(example)
            for n in changed['notes']:
                if n[1]>=1920 and n[3]!=nt.ROLES[role]['channel']:n[4]+=1
            changed['notes'].append(['note',2400,480,0,80,60])
            b=teacher_example(self.model,role,changed)
            self.assertTrue(torch.equal(a['x'],b['x']))
            self.assertTrue(torch.equal(a['labels'],b['labels']))
            valid=a['support'].any(-1)
            self.assertTrue(a['support'][valid].gather(1,a['labels'][valid][:,None]).all())
            self.assertNotIn(self.model.tokenizer.event_ids['key_signature'],a['x'][:,0])
            # The first prediction's prefix ends before its own target note.
            self.assertEqual(int(a['positions'][0]),8)

    def test_windows_project_unsupported_density_without_invented_notes(self):
        e=self.example();e['notes'].append(['note',7680,60,0,70,60])
        self.assertEqual(len(list(windows(e))),1)
        e['notes'].extend([['note',1920+i*30,60,2,48+i,60] for i in range(3)])
        projected=list(windows(e));self.assertEqual(len(projected),1)
        self.assertGreater(projected[0]['projection']['source_notes'],projected[0]['projection']['retained_notes'])
        signatures={(n[1],n[3],n[4],n[5]) for n in e['notes']}
        self.assertTrue(all((n[1],n[3],n[4],n[5]) in signatures for n in projected[0]['notes']))

    def test_cello_projection_monophonic_and_input_unchanged(self):
        source=[['note',0,960,2,40,60],['note',0,960,2,44,60],['note',480,960,2,43,60]]
        original=copy.deepcopy(source);notes=playable_projection(source)
        self.assertEqual(source,original)
        self.assertEqual(notes,[['note',0,480,2,40,60],['note',480,960,2,43,60]])

    def test_missing_instrument_is_not_a_silent_teacher_label(self):
        e=self.example();e['available_roles']=['organ','strings']
        self.assertIsNone(teacher_example(self.model,'cello',e))
        self.assertIsNotNone(teacher_example(self.model,'organ',e))


class CriticTests(unittest.TestCase):
    def test_order_matters_even_with_identical_summary(self):
        torch.manual_seed(9);critic=SequenceCritic()
        a=[['note',0,240,0,64,60],['note',480,240,0,67,60]]
        b=[['note',0,240,0,67,60],['note',480,240,0,64,60]]
        sa=critic_state(nt.Brief(),2,a);sb=critic_state(nt.Brief(),2,b)
        self.assertTrue(torch.equal(sa[:32],sb[:32]))
        self.assertGreater(abs(float((critic(sa)-critic(sb)).detach())),1e-7)

    def test_future_rejected_empty_finite_padding_ignored(self):
        with self.assertRaises(ValueError):critic_state(nt.Brief(),0,[['note',0,240,0,64,60]])
        critic=SequenceCritic();empty=critic_state(nt.Brief(),0,[])
        self.assertTrue(torch.isfinite(critic(empty)))
        a=critic_state(nt.Brief(),1,[['note',0,240,0,64,60]])
        b=a.clone();b[32+FEATURES:-1]=torch.randn((TOKENS-1)*FEATURES)
        self.assertTrue(torch.equal(critic(a),critic(b)))

    def test_sequence_gets_value_learning_gradient(self):
        critic=SequenceCritic();x=critic_state(nt.Brief(),1,[['note',0,240,0,64,60]])
        (critic(x)-1).square().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in critic.parameters()))
        self.assertGreater(float(critic.sequence.weight_ih_l0.grad.abs().sum()),0)


if __name__=='__main__':unittest.main()
