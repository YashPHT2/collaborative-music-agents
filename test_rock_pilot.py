import copy,io,unittest
from types import SimpleNamespace
import torch,mido
import rock_ensemble as rock
from midi_tokenizer import MIDITokenizer
from prepare_rock_pilot import parse,windows

class RockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rock.configure();cls.model=SimpleNamespace(tokenizer=MIDITokenizer('v2'),device=torch.device('cpu'))
    def example(self):
        return dict(notes=[['note',0,480,2,40,80],['note',3840,60,3,38,80],['note',3840,240,0,64,80]],available_roles=['guitar_1','bass','drums'],bpm=120)
    def test_percussion_channel_and_pitch(self):
        w=self.example();ex=rock.teacher(self.model,w,'drums',8)
        event=self.model.tokenizer.tokens2event(ex['labels'][0].tolist())
        decoded=dict(zip(self.model.tokenizer.events['note'],event[1:]))
        self.assertEqual(decoded['channel'],9);self.assertEqual(decoded['pitch'],38)
        self.assertIn(38,rock.ROLES['drums']['pitches']) # not in a C-major melodic mask
    def test_missing_role_mask(self):
        self.assertIsNone(rock.teacher(self.model,self.example(),'keyboard',8))
    def test_no_future_partner_leakage(self):
        a=self.example();b=copy.deepcopy(a);b['notes'][-1][4]=71
        ea=rock.teacher(self.model,a,'drums',8);eb=rock.teacher(self.model,b,'drums',8)
        self.assertTrue(torch.equal(ea['x'],eb['x']))
        self.assertTrue(torch.equal(ea['labels'],eb['labels']))
    def test_sustain_and_stop(self):
        w=self.example();w['notes'].append(['note',3600,960,0,65,80])
        ex=rock.teacher(self.model,w,'guitar_1',9)
        self.assertEqual(ex['kind'],'sustain')
        self.assertEqual(int(ex['labels'][0,0]),self.model.tokenizer.eos_id)
    def test_real_midi_drum_parser(self):
        midi=mido.MidiFile();track=mido.MidiTrack();midi.tracks.append(track)
        for i in range(16):
            track.append(mido.Message('note_on',channel=9,note=38,velocity=80,time=120))
            track.append(mido.Message('note_off',channel=9,note=38,velocity=0,time=120))
        f=io.BytesIO();midi.save(file=f);notes,available,bpm=parse(f.getvalue())
        self.assertIn('drums',available);self.assertEqual(len(notes),16)
        self.assertTrue(all(n[3]==3 and n[4]==38 for n in notes))

if __name__=='__main__':unittest.main()
