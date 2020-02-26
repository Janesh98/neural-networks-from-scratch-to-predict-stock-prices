import unittest
import sys
import numpy as np
# to import from a parent directory
sys.path.append('../')
from NeuralNetwork import NeuralNetwork, mse, rmse, mape

class MseTestCase(unittest.TestCase):
    def test_mse(self):
        # if both arrays are the same the mean squared error should = 0
        self.assertEqual(mse(np.arange(5), np.arange(5)), 0.0)

class NeuralNetworkTestCase(unittest.TestCase):
    def test_NeuralNetwork(self):
        train_input = [[100, 100] for i in range(100)]
        train_target = [[100] for i in range(100)]

        test_input = [[101, 101] for i in range(50)]
        test_target = [[101]for i in range(50)]

        # normalize
        train_input = np.array(train_input) / 1000
        train_target = np.array(train_target) / 1000
        test_input = np.array(test_input) / 1000 

        test_target = np.array(test_target)       

        NN = NeuralNetwork()

        # number of training cycles
        epochs = 100

        # train the neural network
        for e in range(epochs):
            for p in train_input:
                train_output = NN.train(train_input, train_target)

        # de-normalize
        train_output *= 1000
        train_target *= 1000

        test_output = NN.test(test_input)

        # de-normalize
        test_output *= 1000

        self.assertGreaterEqual(100 - mape(train_target, train_output), 99.99)
        self.assertGreaterEqual(100 - mape(test_target, test_output), 98.00)

if __name__ == '__main__':
    unittest.main()