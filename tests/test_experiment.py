import json
import tempfile
import unittest
from pathlib import Path
from scripts.run_experiment import metric_ready, generations_ready, latest_checkpoint


class ExperimentTests(unittest.TestCase):
    def test_baseline_requires_matching_full_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'baseline.json'
            data = dict(identity={'revision':'a'},adapter=None,split='validation',target_tokens=10)
            path.write_text(json.dumps(data))
            self.assertTrue(metric_ready(path,{'revision':'a'}))
            self.assertFalse(metric_ready(path,{'revision':'b'}))
            data['split']='intermediate'
            path.write_text(json.dumps(data))
            self.assertFalse(metric_ready(path,{'revision':'a'}))

    def test_incomplete_generation_suite_is_not_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'generation-settings-base.json').write_text(json.dumps({'identity':{},'adapter':None}))
            rows=[dict(kind=kind,generation='text') for kind in ['style']*20+['general']*20]
            output=root/'generations-base.jsonl'
            output.write_text('\n'.join(json.dumps(row) for row in rows))
            self.assertTrue(generations_ready(root,{}))
            output.write_text('\n'.join(json.dumps(row) for row in rows[:-1]))
            self.assertFalse(generations_ready(root,{}))

    def test_partial_checkpoint_is_skipped(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for step in [50,100]:
                checkpoint=root/'checkpoints'/f'checkpoint-{step}'
                checkpoint.mkdir(parents=True)
                for name in ['trainer_state.json','adapter_model.safetensors','optimizer.pt','scheduler.pt','rng_state.pth']:
                    (checkpoint/name).touch()
            (root/'checkpoints/checkpoint-50/complete.json').write_text('{"step":50}')
            self.assertEqual(latest_checkpoint(root).name,'checkpoint-50')
