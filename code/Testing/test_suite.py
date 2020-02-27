import unittest
from tests import MseTestCase, NeuralNetworksTestCase

def my_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(MseTestCase))
    suite.addTest(unittest.makeSuite(NeuralNetworksTestCase))
    runner = unittest.TextTestRunner()
    print(runner.run(suite))

if __name__ == "__main__":
    my_suite()