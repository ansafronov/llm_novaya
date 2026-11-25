import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from novaya.tracking import setup, sanitized, safe_argv


class TrackingTests(unittest.TestCase):
    def fake_wandb(self):
        fake = Mock()
        fake.init.return_value.id = 'test-run'
        return fake

    def test_secret_redaction_preserves_token_counts(self):
        self.assertEqual(sanitized({'api_key':'secret','nested':{'hf_token':'secret'},'max_new_tokens':100}),
                         {'api_key':'[redacted]','nested':{'hf_token':'[redacted]'},'max_new_tokens':100})
        self.assertEqual(safe_argv(['python','--api-key=secret','--hf-token','secret','--max-new-tokens','10']),
                         ['python','--api-key=[redacted]','--hf-token','[redacted]','--max-new-tokens','10'])

    def test_token_is_not_in_public_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            token = Path(folder)/'token'
            token.write_text('test-secret-value\n')
            fake = Mock()
            fake.init.return_value.id = 'test-run'
            config = {'output':folder,'model':'test','wandb':{'token_file':str(token),'enabled':True}}
            with patch.dict(sys.modules, {'wandb':fake}), patch.dict(os.environ, {}, clear=True):
                setup(config, 'smoke')
                self.assertEqual(os.environ['WANDB_API_KEY'], 'test-secret-value')
                kwargs = fake.init.call_args.kwargs
                self.assertNotIn('wandb', kwargs['config'])
                self.assertNotIn('test-secret-value', str(kwargs))
                self.assertFalse(kwargs['save_code'])
                snapshot = next((Path(folder)/'launches').rglob('launch-parameters.json')).read_text()
                self.assertNotIn('test-secret-value',snapshot)
                fake.init.return_value.log_artifact.assert_called_once()

    def test_disabled_does_not_import_wandb(self):
        self.assertIsNone(setup({'wandb':{'enabled':False}},'test'))

    def test_offline_needs_no_token(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(sys.modules, {'wandb':self.fake_wandb()}), patch.dict(os.environ, {'WANDB_MODE':'offline'}, clear=True):
            setup({'output':folder},'test')
            self.assertNotIn('WANDB_API_KEY',os.environ)
