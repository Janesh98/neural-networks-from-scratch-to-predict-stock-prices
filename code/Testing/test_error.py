import unittest
import numpy as np
import sys
# to import from a parent directory
sys.path.append('../')
from utils import mse, rmse, mape

class ErrorTestCase(unittest.TestCase):
    """ if both arrays are the same the error should be 0 """

    def test_mse(self):
        test = np.array([10, 10, 10, 10, 10])
        self.assertEqual(mse(test, test), 0.0)

    def test_rmse(self):
        test = np.array([10, 10, 10, 10, 10])
        self.assertEqual(rmse(test, test), 0.0)

    def test_mape(self):
        test = np.array([10, 10, 10, 10, 10])
        self.assertEqual(mape(test, test), 0.0)

if __name__ == '__main__':
    unittest.main()