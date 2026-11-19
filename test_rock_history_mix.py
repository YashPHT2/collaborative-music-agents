import contextlib,copy,io,unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
import rock_history_mix as m
from midi_tokenizer import MIDITokenizer

class Planner(torch.nn.Module):
    def __init__(self):super().__init__();self.p=torch.nn.Parameter(torch.zeros(1));self.inputs=[]
    def forward(self,x,hidden=None):self.inputs.append(x.detach().clone());return torch.full((1,1,128),.25),hidden

class Tests(unittest.TestCase):
    def setUp(self):
        m.rock.configure();self.model=SimpleNamespace(device=torch.device('cpu'),tokenizer=MIDITokenizer('v2'))
        self.w=dict(id='source',group='group',split='train',bpm=120,available_roles=['guitar_1'],notes=[['note',0,120,0,60,80],['note',2400,120,0,62,80],['note',3840,120,0,64,80]])
    def record(self):
        return dict(source_window='source',group='group',split='train',notes=[self.w['notes'][0],['note',1920,3840,0,64,80]])
    def test_cold_start_and_future_rejection(self):
        self.assertIsNone(m.rock.teacher(self.model,self.w,'guitar_1',0))
        self.assertIsNotNone(m.rock.teacher(self.model,self.w,'guitar_1',0,allow_warmup=True))
        with self.assertRaises(ValueError):m.rock.teacher(self.model,self.w,'guitar_1',8,history_override=[self.w['notes'][-1]])
    def test_unsupported_targets_fall_back_without_relabelling(self):
        _,real,_=m.examples(self.model,self.w,'guitar_1');_,aug,stats=m.examples(self.model,self.w,'guitar_1',self.record())
        self.assertGreater(stats['fallback'],0);self.assertTrue(torch.equal(real[0]['labels'],aug[0]['labels']))
        self.assertFalse(aug[0]['history_mixed'])
        self.assertTrue(torch.equal(real[0]['x'],aug[0]['x']))
    def test_missing_role_and_split_guard(self):
        cold,cont,_=m.examples(self.model,self.w,'drums',self.record());self.assertEqual(cold+cont,[])
        wrong=self.record();wrong['split']='test'
        with self.assertRaises(ValueError):m.mixed_ledger(self.w,wrong)
    def test_future_record_and_overlap_guard(self):
        wrong=self.record();wrong['notes']=[['note',3840,60,0,60,80]]
        with self.assertRaises(ValueError):m.mixed_ledger(self.w,wrong)
        self.assertFalse(m.supported_history([['note',0,480,0,60,80],['note',120,120,0,60,80]]))
    def test_rollin_never_reads_replaced_or_future_target_notes(self):
        calls=[]
        class Sampler:
            def __init__(self,model,role,seed,brief,beat,history):self.beat=beat;calls.append((beat,copy.deepcopy(history)))
            def propose(self,actor,state):return [['note',self.beat*480,60,0,70,80]],[]
        with patch.object(m.rock,'Sampler',Sampler),contextlib.redirect_stdout(io.StringIO()):
            record=m.rollin(self.model,SimpleNamespace(activate=lambda f:None),Planner(),lambda *a:torch.zeros(1024),{'guitar_1':None},self.w,[.2,.5,.2,.5],1)
        self.assertEqual([b for b,h in calls],[4,5,6,7])
        self.assertTrue(all(n[4] not in [62,64] and n[1]<b*480 for b,h in calls for n in h))
        self.assertEqual(len(record['generated']),4)
    def test_planner_falls_back_on_invalid_hybrid_past(self):
        a=Planner();b=Planner();goal=torch.tensor([.2,.5,.2,.5])
        m.plan_sequence(a,self.w,goal,self.record());m.plan_sequence(b,self.w,goal)
        for x,y in zip(a.inputs[4:],b.inputs[4:]):self.assertTrue(torch.equal(x[:,:,128:256],y[:,:,128:256]))
    def test_bank_never_crosses_requested_split(self):
        validation={**self.w,'id':'v','group':'v','split':'validation'}
        selected=m.select_windows([self.w,validation],'train');self.assertEqual([w['id'] for w in selected],['source'])
    def test_selection_rejects_attribute_regression(self):
        from train_rock_history_mix import eligible
        baseline={task:{r:dict(event=1.,attributes=1.) for r in m.rock.ROLES} for task in ['startup','real','mixed']}
        candidate=copy.deepcopy(baseline)
        for task in candidate:
            for r in candidate[task]:candidate[task][r]['event']=.9
        self.assertTrue(eligible(candidate,baseline))
        candidate['mixed']['drums']['attributes']=1.16
        self.assertFalse(eligible(candidate,baseline))

if __name__=='__main__':unittest.main()
