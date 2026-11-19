import unittest
import random
import json
import tempfile
from pathlib import Path
import torch
import ensemble_repair as er
import offline_ensemble as oe


class RepairTests(unittest.TestCase):
    def test_labels_include_sustain_and_missing(self):
        w=dict(available_roles=['organ'],notes=[['note',0,960,0,60,64]],bars=8)
        self.assertEqual(er.beat_kind(w,'organ',0),'onset')
        self.assertEqual(er.beat_kind(w,'organ',1),'sustain')
        self.assertEqual(er.beat_kind(w,'organ',2),'silence')
        self.assertEqual(er.beat_kind(w,'cello',0),'missing')
        w['source_roles']=['organ','cello']
        self.assertEqual(er.beat_kind(w,'cello',0),'excluded_low_support')

    def test_upgrade_preserves_original_predictions(self):
        torch.manual_seed(1)
        oldp,oldc=oe.Planner(),oe.Conditions()
        p,c=er.upgrade(dict(planner=oldp.state_dict(),conditions=oldc.state_dict()),'cpu')
        x=torch.rand(1,3,65);expanded=torch.zeros(1,3,97)
        expanded[:,:,:32]=x[:,:,:32];expanded[:,:,48:80]=x[:,:,32:64];expanded[:,:,-1]=x[:,:,-1]
        self.assertTrue(torch.allclose(oldp(x)[0],p(expanded)[0][:,:,:32],atol=1e-6))
        plan=torch.rand(32);features=torch.zeros(1,8)
        self.assertTrue(torch.equal(oldc(plan,features,'organ'),c(torch.cat([plan,torch.zeros(16)]),features,'organ')))

    def test_sequence_uses_the_same_causal_scheduler(self):
        p=er.Planner();notes=[['note',0,480,0,60,64],['note',4000,120,1,67,64]]
        sequence=er.plan_sequence(p,notes,8)
        parts=[];prev=hidden=None
        for start in range(0,32,8):
            plan,prev,hidden=er.plan_segment(p,[n for n in notes if n[1]<start*480],start,8,prev,hidden)
            parts.extend(plan)
        self.assertTrue(torch.equal(sequence,torch.stack(parts)))
        altered=notes+[['note',15000,60,2,48,64]]
        self.assertTrue(torch.equal(sequence[:6],er.plan_sequence(p,altered,8)[:6]))

    def test_profile_only_changes_density_register_velocity(self):
        p=er.Planner();q={str(i):[.35,.65] for i in range(12,21)}
        a,_,_=er.plan_segment(p,[],0,8)
        b,_,_=er.plan_segment(p,[],0,8,profile='warm_build',quantiles=q)
        # First bar pitch and rhythm are untouched; later predictions may respond causally.
        self.assertTrue(torch.equal(a[0][:12],b[0][:12]))
        self.assertTrue(torch.equal(a[0][21:],b[0][21:]))
        self.assertTrue(torch.allclose(b[0][12:21],torch.full((9,),.35)))

    def test_gate_rejects_one_role_regression(self):
        initial={r:dict(event=1.,attributes=1.) for r in er.ROLES}
        good={r:dict(event=.9,attributes=1.01) for r in er.ROLES}
        self.assertTrue(er.eligible(good,initial))
        good['strings']['attributes']=1.03
        self.assertFalse(er.eligible(good,initial))

    def test_sounding_coverage_not_note_count(self):
        stats=er.participation([['note',0,3840,1,60,64]])
        self.assertEqual(stats['strings']['sounding_beats'],8)
        self.assertEqual(stats['strings']['notes'],1)
        self.assertEqual(stats['organ']['sounding_beats'],0)

    def test_loss_aggregation_balances_strata_and_compositions(self):
        rows=[dict(group='a',kind='silence',losses={'first_event':0.}) for _ in range(20)]
        rows += [dict(group='a',kind='onset',losses={'first_event':2.}),dict(group='b',kind='onset',losses={'first_event':3.})]
        self.assertEqual(er.aggregate(rows)['event'],2.)

    def test_partner_draft_guards(self):
        with self.assertRaises(ValueError):oe.draft_features([['note',0,60,0,60,64]],'organ',0)
        with self.assertRaises(ValueError):oe.draft_features([['note',3840,60,1,60,64]],'organ',0)

    def test_masked_loss_gradients_are_finite(self):
        class Model(torch.nn.Module):
            device='cpu'
            def __init__(self):
                super().__init__();self.logits=torch.nn.Parameter(torch.zeros(2,8,12))
            def forward(self,x):return torch.zeros(1,2,3)
            def forward_token(self,hidden,y):return self.logits
        model=Model();support=torch.zeros(2,8,12,dtype=torch.bool)
        support[:,0,:2]=True
        for col in [2,5,6,7]:support[0,col,:4]=True
        ex=dict(x=torch.zeros(2,8,dtype=torch.long),positions=torch.tensor([0,1]),
                labels=torch.zeros(2,8,dtype=torch.long),support=support)
        losses=er.category_losses(model,ex)
        self.assertEqual(set(losses),set(er.CATEGORIES))
        sum(losses.values()).backward()
        self.assertTrue(torch.isfinite(model.logits.grad).all())
        self.assertEqual(float(model.logits.grad[1,7].abs().sum()),0.)
        model.zero_grad()
        with torch.no_grad():model.logits[0,0,1]=100.
        er.category_losses(model,ex,training=True)['first_event'].backward()
        self.assertGreater(abs(float(model.logits.grad[0,0,1])),1.)

    def test_missing_roles_skip_teacher_without_model_access(self):
        self.assertIsNone(er.teacher(None,None,dict(available_roles=['organ']),'cello',0,None))

    def test_right_padded_batch_matches_serial_losses_and_gradients(self):
        class Model(torch.nn.Module):
            device='cpu'
            tokenizer=type('Tokenizer',(),{'pad_id':0})()
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.arange(96,dtype=torch.float32).reshape(8,12)/100)
            def forward(self,x):
                # Causal states make any padding leakage observable.
                return x.float().cumsum(1)
            def forward_token(self,hidden,y,cache=None):
                return (hidden @ self.weight)[:,None,:].expand(-1,8,-1)
        model=Model();examples=[]
        for length in (2,5):
            support=torch.zeros(1,8,12,dtype=torch.bool);support[0,0,:2]=True;support[0,5,:4]=True
            examples.append(dict(x=torch.ones(length,8,dtype=torch.long),positions=torch.tensor([length-1]),
                                 labels=torch.zeros(1,8,dtype=torch.long),support=support))
        delta=torch.ones(8)
        serial=[er.category_losses(oe.ConditionedModel(model,delta),ex,True) for ex in examples]
        sum(sum(r.values()) for r in serial).backward();expected=model.weight.grad.clone();model.zero_grad()
        batch=er.batch_category_losses(model,examples,[delta,delta],True)
        sum(sum(r.values()) for r in batch).backward()
        self.assertTrue(torch.allclose(expected,model.weight.grad,atol=1e-5))
        for a,b in zip(serial,batch):
            for key in a:self.assertTrue(torch.allclose(a[key],b[key],atol=1e-6))

    def test_rating_identity_completeness_and_duplicate_guard(self):
        from score_repair_ratings import score
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'panel_identity.json').write_text(json.dumps(dict(sha256='abc')))
            ballot=dict(panel_sha256='abc',ratings={c:{d:3 for d in ['enjoyment','development','ensemble_fit','warmth']}
                                                    for c in ['baseline','repaired','warm_build']})
            path=root/'ballot.json';path.write_text(json.dumps(ballot))
            result=score(root,[path]);self.assertFalse(any(result['listening_gate'].values()))
            with self.assertRaises(ValueError):score(root,[path,path])
            ballot['panel_sha256']='other';path.write_text(json.dumps(ballot))
            with self.assertRaises(ValueError):score(root,[path])


if __name__=='__main__':unittest.main()
