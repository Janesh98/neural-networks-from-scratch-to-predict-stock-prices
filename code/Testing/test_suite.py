import unittest
from tests import ErrorTestCase, NeuralNetworksTestCase

def my_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(ErrorTestCase))
    suite.addTest(unittest.makeSuite(NeuralNetworksTestCase))
    runner = unittest.TextTestRunner()
    print(runner.run(suite))

if __name__ == "__main__":
    my_suite()