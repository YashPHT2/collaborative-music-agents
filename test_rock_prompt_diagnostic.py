import contextlib,io,unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
import rock_prompt_diagnostic as d

class Planner(torch.nn.Module):
    def __init__(self):super().__init__();self.p=torch.nn.Parameter(torch.zeros(1));self.inputs=[]
    def forward(self,x,hidden=None):
        self.inputs.append(x.detach().clone());return torch.full((1,1,128),.25),hidden

class Tests(unittest.TestCase):
    def setUp(self):
        d.g.rock.configure();self.prefix=[['note',0,480,i,38 if i==3 else 60,80] for i in range(5)]
        self.protocol=dict(prefix=self.prefix,goal=[.2,.5,.2,.5],bpm=120,seed=9976201)
    def test_reject_future_or_crossing_prompt(self):
        for notes in [[['note',3840,60,0,60,80]],[['note',3780,120,0,60,80]]]:
            with self.assertRaises(ValueError):d.check_prefix(notes)
    def run_fake(self,condition):
        calls=[]
        class Sampler:
            def __init__(self,model,role,seed,brief,beat,history):calls.append((beat,history))
            def propose(self,actor,state):return [],[]
        planner=Planner();adapters=SimpleNamespace(activate=lambda f:None)
        with patch.object(d.g.rock,'Sampler',Sampler),contextlib.redirect_stdout(io.StringIO()):
            ep=d.compose(SimpleNamespace(device=torch.device('cpu')),adapters,planner,lambda *a:torch.zeros(1024),{r:None for r in d.g.rock.ROLES},self.protocol,condition)
        return ep,calls,planner
    def test_reference_attribution_and_no_future_access(self):
        ep,calls,_=self.run_fake('prompted')
        self.assertEqual(ep['notes'],self.prefix);self.assertEqual(ep['decisions'],[])
        self.assertTrue(all(beat>=8 and all(n[1]<beat*480 for n in ns) for beat,ns in calls))
        self.assertEqual(len(calls),16*5)
        plain,calls,_=self.run_fake('unprompted');self.assertEqual(plain['notes'],[])
        self.assertEqual(len(calls),24*5)
    def test_training_generation_plan_alignment(self):
        _,_,inference=self.run_fake('prompted');training=Planner()
        d.g.l.plan_sequence(training,self.prefix,6,list(d.g.rock.ROLES),torch.tensor(self.protocol['goal']))
        self.assertEqual(len(inference.inputs),6)
        for a,b in zip(inference.inputs,training.inputs):self.assertTrue(torch.equal(a,b))

if __name__=='__main__':unittest.main()
