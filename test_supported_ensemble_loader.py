import copy
import unittest
from unittest.mock import patch
import torch

import supported_ensemble_loader as l
import ensemble_repair as er


class SupportedLoaderTests(unittest.TestCase):
    def window(self):
        return dict(notes=[['note',3360,960,0,60,64],['note',3900,480,1,67,64]],
                    loss_start_beat=8,loss_end_beat=24,available_roles=['organ','strings','cello'],bars=6)

    def test_prefix_and_missing_roles_never_call_teacher(self):
        w=self.window()
        with patch.object(l.se,'teacher_example',side_effect=AssertionError('must not call')):
            self.assertIsNone(l.teacher(None,w,'organ',7))
            w['available_roles'].remove('cello')
            self.assertIsNone(l.teacher(None,w,'cello',8))

    def test_history_contains_supported_sustain_but_no_current_partner_target(self):
        w=self.window()
        with patch.object(l.se,'teacher_example',return_value=None) as teacher:
            l.teacher(None,w,'organ',8)
            history=teacher.call_args.kwargs['history']
            self.assertEqual(history,[w['notes'][0]])
            self.assertFalse(any(n[1]>=3840 for n in history))

    def test_category_and_active_balance_and_transfer_weight(self):
        def example(kind):
            support=torch.zeros(1,8,3,dtype=torch.bool);support[0,0,:2]=True
            return dict(support=support,kind=kind)
        examples=[example('onset'),example('onset'),example('silence')]
        weights=l.balance_weights(examples,role_weight=.35,normalizer=3)
        self.assertAlmostEqual(sum(w['first_event'] for w in weights),.35/3)
        self.assertAlmostEqual(weights[0]['first_event']+weights[1]['first_event'],weights[2]['first_event'])

    def test_provenance_regression_blocks_average_improvement(self):
        def score(event=1.,attributes=1.):return dict(event=event,attributes=attributes)
        baseline={r:dict(all=score(),exact=score(),transfer=score()) for r in er.ROLES}
        candidate=copy.deepcopy(baseline)
        for r in er.ROLES:candidate[r]['all']=score(.9,.99)
        self.assertTrue(l.selection_eligible(candidate,baseline))
        candidate['organ']['exact']=score(1.,1.03)
        self.assertFalse(l.selection_eligible(candidate,baseline))


if __name__=='__main__':unittest.main()
