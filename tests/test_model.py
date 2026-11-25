import importlib.util
import math
import unittest
from novaya.model import Collator, score
from test_data import Tokenizer


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch not installed')
class ModelTests(unittest.TestCase):
    def test_padding_labels(self):
        tok = Tokenizer()
        tok.pad_token_id = 0
        rows = [dict(input_ids=[1,2], attention_mask=[1,1], labels=[-100,2]),
                dict(input_ids=[3], attention_mask=[1], labels=[3])]
        batch = Collator(tok)(rows)
        self.assertEqual(batch['labels'].tolist(), [[-100,2],[3,-100]])
        self.assertEqual(batch['attention_mask'].tolist(), [[1,1],[1,0]])

    def test_score_uses_token_sums_and_shift(self):
        import torch
        from types import SimpleNamespace
        class UniformModel:
            device = 'cpu'
            def eval(self):
                pass
            def __call__(self, input_ids, attention_mask):
                return SimpleNamespace(logits=torch.zeros((*input_ids.shape,100)))
        tok = Tokenizer()
        tok.pad_token_id = 0
        from unittest.mock import patch
        with patch('torch.cuda.max_memory_allocated', return_value=0):
            result = score(UniformModel(), tok, [{'title':'A','text':'abc'}, {'title':'B','text':'abcde'}],20)
        self.assertEqual(result['target_tokens'],10)
        self.assertAlmostEqual(result['nll_sum'],10*math.log(100),places=4)
        self.assertAlmostEqual(result['ppl'],100,places=4)
