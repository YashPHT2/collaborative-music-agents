"""Tests for shared-context visibility, causal phrase exchange, and role identity."""
import unittest
from unittest.mock import patch

from collaborative_duet import BAR,ROLES,compose_bar,visible_history,role_seed,score,validate_export,signature


class FakeSampler:
    instances=[]

    def __init__(self,model,role,seed,bpm,temperature,history):
        self.role=role
        self.observed=list(history)
        self.calls=[]
        self.committed=[]
        self.__class__.instances.append(self)

    def append_note(self,note,track):
        self.observed.append(note)

    def commit(self,note):
        self.committed.append(note)

    def sample_between(self,left,right):
        self.calls.append(dict(left=left,right=right,observed=list(self.observed)))
        # One sample at the start of each beat window, then WAIT.
        if any(n[1]>=left and n[1]<right for n in self.committed):
            return None
        return ['note',left,min(480,right-left),ROLES[self.role]['channel'],
                72 if self.role=='melody' else 48,80]


class DuetTests(unittest.TestCase):
    def setUp(self):
        FakeSampler.instances=[]

    def test_shared_history_contains_partner_but_not_current_or_future_bars(self):
        ledger=[['note',0,480,0,72,80],['note',0,480,1,48,60],
                ['note',BAR,480,1,50,60],['note',BAR*2,480,0,74,80]]
        shared=visible_history(ledger,'melody',BAR,True)
        private=visible_history(ledger,'melody',BAR,False)
        self.assertEqual({n[3] for n in shared},{0,1})
        self.assertEqual({n[3] for n in private},{0})
        self.assertTrue(all(n[1]<BAR for n in shared))
        shared[0][4]=99
        self.assertEqual(ledger[0][4],72)

    @patch('collaborative_duet.RoleSampler',FakeSampler)
    def test_accompaniment_sees_melody_one_onset_at_a_time(self):
        lead=[['note',0,420,0,72,90],['note',480,420,0,78,90]]
        notes,audit=compose_bar(None,'accompaniment',[],0,BAR,1,100,0.9,incoming=lead)
        sampler=FakeSampler.instances[0]
        self.assertEqual([n[4] for n in sampler.calls[0]['observed']],[72])
        second_window=next(call for call in sampler.calls if call['left']>=480)
        self.assertEqual([n[4] for n in second_window['observed']],[72,78])
        self.assertEqual(audit['incoming_events'],2)
        self.assertTrue(all(n[3]==1 for n in notes))

    @patch('collaborative_duet.RoleSampler',FakeSampler)
    def test_melody_advances_past_its_note_end(self):
        notes,_=compose_bar(None,'melody',[],0,BAR,1,100,0.9)
        self.assertGreater(len(notes),0)
        self.assertTrue(all(a[1]+a[2]<=b[1] for a,b in zip(notes,notes[1:])))

    @patch('collaborative_duet.RoleSampler',FakeSampler)
    def test_pre_onset_rest_does_not_expose_future_melody_pitch(self):
        lead=[['note',480,420,0,78,90]]
        compose_bar(None,'accompaniment',[],0,BAR,1,100,0.9,incoming=lead)
        sampler=FakeSampler.instances[0]
        self.assertEqual(sampler.calls[0]['observed'],[])

    def test_role_random_streams_are_deterministic_and_separate(self):
        values=[role_seed(2027,r,b) for r in ROLES for b in range(8)]
        self.assertEqual(len(set(values)),16)
        self.assertEqual(role_seed(2027,'melody',2),role_seed(2027,'melody',2))

    def test_separate_tracks_and_isolation_preserve_generated_notes(self):
        import tempfile
        from pathlib import Path
        from generate_piano import MIDI
        ledger=[['note',0,420,0,72,80],['note',0,480,1,48,60],
                ['note',480,420,0,74,80],['note',480,480,1,52,60]]
        with tempfile.TemporaryDirectory() as directory:
            for channel in [None,0,1]:
                path=Path(directory)/'duet.mid'
                path.write_bytes(MIDI.score2midi(score(ledger,100,channel)))
                expected=ledger if channel is None else [n for n in ledger if n[3]==channel]
                result=validate_export(path,expected,BAR,100)
                self.assertTrue(result['note_events_preserved'])


if __name__=='__main__':
    unittest.main()
