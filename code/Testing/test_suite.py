import unittest
from test_error import ErrorTestCase
from test_NeuralNetworks import NeuralNetworksTestCase
from test_normalize import NormalizeTestCase

def my_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(ErrorTestCase))
    suite.addTest(unittest.makeSuite(NeuralNetworksTestCase))
    suite.addTest(unittest.makeSuite(NormalizeTestCase))
    runner = unittest.TextTestRunner()
    print(runner.run(suite))

if __name__ == "__main__":
    my_suite()