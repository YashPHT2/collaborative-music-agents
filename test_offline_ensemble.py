import copy, unittest, tempfile, json
from pathlib import Path
import torch
import offline_ensemble as oe
import note_trio as nt
import ensemble_data as ed
from score_ensemble_ratings import score

class OfflineTests(unittest.TestCase):
    def test_extended_durations_preserve_legacy_default(self):
        raw=[dict(start=0,duration=3840,pitch=60,velocity=64,program=19)]
        self.assertEqual(ed.canonicalize(raw)[0]['organ'][0][2],1920)
        self.assertEqual(ed.canonicalize(raw,max_duration=3840)[0]['organ'][0][2],3840)
    def test_own_drafts_and_unannounced_future_rejected(self):
        with self.assertRaises(ValueError):oe.draft_features([['note',0,480,0,60,64]],'organ',0)
        with self.assertRaises(ValueError):oe.draft_features([['note',3840,480,1,60,64]],'organ',0)
        with self.assertRaises(ValueError):oe.compose(None,None,None,None,None,prefix=[['note',1920,480,1,60,64]])
    def test_draft_encoder_is_permutation_invariant_at_input(self):
        ns=[['note',480,480,1,60,64],['note',0,480,2,48,64]]
        self.assertTrue(torch.equal(oe.draft_features(ns,'organ',0),oe.draft_features(ns[::-1],'organ',0)))
    def test_description_respects_bar(self):
        ns=[['note',0,480,0,60,64],['note',1920,480,1,67,64]]
        self.assertEqual(float(oe.description(ns,0)[7]),0)
        self.assertEqual(float(oe.description(ns,0)[0]),1)
    def test_limits_restore_on_error(self):
        old=copy.deepcopy(nt.ROLES)
        with self.assertRaises(RuntimeError):
            with oe.limits():
                self.assertEqual(nt.ROLES['strings']['cap'],4);raise RuntimeError()
        self.assertEqual(nt.ROLES,old)
    def test_conditioning_starts_neutral_and_learns(self):
        c=oe.Conditions();p=torch.zeros(32);d=oe.draft_features([],'organ',0)
        self.assertEqual(c(p,d,'organ').abs().sum().item(),0)
        opt=torch.optim.Adam(c.parameters(),lr=.01)
        loss=(c(p,d,'organ')-1).square().mean();loss.backward();opt.step()
        self.assertGreater(c(p,d,'organ').abs().sum().item(),0)
    def test_planner_backward_and_state(self):
        p=oe.Planner();y,h=p(torch.zeros(1,8,65));self.assertEqual(y.shape,(1,8,32))
        y.sum().backward();self.assertIsNotNone(p.head.weight.grad)
        q=oe.Planner();q.load_state_dict(p.state_dict());self.assertTrue(torch.equal(y,q(torch.zeros(1,8,65))[0]))
    def test_cross_split_window_duplicates_excluded(self):
        ns=[['note',i*480,480,0,60+i%4,64] for i in range(32)]
        records=[dict(song=s,group=s,split=s,notes=ns) for s in ['train','validation']]
        ws=oe.candidates(records);self.assertEqual({w['split'] for w in ws},{'validation'})

    def test_listening_improvement_does_not_override_organ_regression(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);panel=[dict(id=f'{m}_{i}',mode=m) for m in ['baseline','planned','revised'] for i in range(4)]
            (root/'panel_key.json').write_text(json.dumps(panel));(root/'panel_identity.json').write_text('{"sha256":"test"}')
            (root/'data_manifest.json').write_text('{"gate_passed":true}')
            probes=[dict(role=r,song=str(i),baseline_correct=.7,scores=dict(correct=.6 if r=='organ' else .8,missing=.5,mismatched=.5)) for r in oe.ROLES for i in range(10)]
            (root/'response_probes.json').write_text(json.dumps(probes));files=[]
            for voter in range(3):
                path=root/f'{voter}.json';files.append(path)
                ratings={p['id']:{k:2 if p['mode']=='baseline' else 4 for k in ['enjoyment','development','ensemble_fit']} for p in panel}
                path.write_text(json.dumps(dict(panel_sha256='test',ratings=ratings)))
            result=score(root,files);self.assertTrue(result['listening_gate_passed'])
            self.assertFalse(json.loads((root/'acceptance.json').read_text())['passed'])

if __name__=='__main__':unittest.main()
