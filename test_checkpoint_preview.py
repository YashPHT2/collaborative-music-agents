import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch
import orchestra_learning as l
from evaluate_trained_orchestra import comparison_inputs


class SnapshotPreviewTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.initial=self.root/'initial.pt';self.trained=self.root/'phrases.pt'
        self.base=dict(format_version=4,base_sha256='frozen',roles=l.ROLES,families=l.FAMILIES)
        torch.save(self.base,self.initial)
        self.saved={**self.base,'h100_config':dict(init_sha256=hashlib.sha256(self.initial.read_bytes()).hexdigest())}
        torch.save(self.saved,self.trained)
        baseline={r:dict(event=1.,attributes=1.) for r in l.ROLES}
        self.log=dict(complete=False,stage='phrases',step=154,selected_step=128,selections={},initial=baseline,
                      validations=[dict(stage='phrases',step=128,passed=True,
                                        scores={r:dict(event=.8,attributes=.9) for r in l.ROLES})])
        self.write_log()

    def write_log(self):
        (self.root/'training_log.json').write_text(json.dumps(self.log))

    def test_in_progress_snapshot_keeps_incomplete_status(self):
        result=comparison_inputs(self.root,self.trained,self.initial)
        self.assertTrue(result[5]);self.assertFalse(result[6]['training_complete'])
        self.assertEqual(result[6]['log_selected_step'],128)

    def test_normal_evaluation_still_requires_completed_training(self):
        with self.assertRaisesRegex(ValueError,'not finished'):comparison_inputs(self.root)

    def test_note_attribute_regression_does_not_pass(self):
        self.log['validations'][0]['scores']['cellos']['attributes']=1.1;self.write_log()
        self.assertFalse(comparison_inputs(self.root,self.trained,self.initial)[5])

    def test_wrong_initialization_is_rejected(self):
        wrong=self.root/'different.pt';torch.save({**self.base,'extra':'different weights'},wrong)
        with self.assertRaisesRegex(ValueError,'not the initialization'):
            comparison_inputs(self.root,self.trained,wrong)


if __name__=='__main__':unittest.main()
