import unittest
from tests import MseTestCase

def my_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(MseTestCase))
    runner = unittest.TextTestRunner()
    print(runner.run(suite))

if __name__ == "__main__":
    my_suite()