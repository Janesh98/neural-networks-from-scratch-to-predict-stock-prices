import unittest
import numpy as np
import sys
# to import from a parent directory
sys.path.append('../')
from normalize import Normalize

class NormalizeTestCase(unittest.TestCase):
    """ ensure data is normalized between 0 and 1, 
        then reverted back to the original values """
    
    def test_normalize(self):
        test = np.arange(1000)

        # normalize
        scaler = Normalize(test)
        normalized = scaler.normalize_data(test)

        min_val = min(normalized)
        max_val = max(normalized)

        # ensure values scaled to range (0, 1)
        self.assertGreaterEqual(min_val, 0.0)
        self.assertLessEqual(max_val, 1.0)

        # denormalize
        denormalized = scaler.denormalize_data(normalized)

        # ensure denormalized values are the same as the original
        for x, y in zip(test, denormalized):
            try:
                self.assertEqual(x, y)
            except AssertionError:
                self.assertAlmostEqual(x, y, 12)

if __name__ == '__main__':
    unittest.main()