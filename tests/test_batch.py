import unittest
from novaya.batch import accumulation_for


class BatchTests(unittest.TestCase):
    def test_effective_batch_is_preserved(self):
        self.assertEqual(accumulation_for(1,16),16)
        self.assertEqual(accumulation_for(2,16),8)
        self.assertEqual(accumulation_for(4,16),4)

    def test_nondivisible_batch_is_rejected(self):
        for micro,effective in [(3,16),(0,16),(2,0)]:
            with self.assertRaises(ValueError):
                accumulation_for(micro,effective)
