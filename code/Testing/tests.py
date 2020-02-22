import unittest
import sys
# to import from a parent directory
sys.path.append('../')
from NeuralNetwork import mse
import numpy as np

class MseTestCase(unittest.TestCase):
    def test_mse(self):
        # if both arrays are the same the mean squared error should = 0
        self.assertEqual(mse(np.arange(5), np.arange(5)), 0.0)

if __name__ == '__main__':
    unittest.main()