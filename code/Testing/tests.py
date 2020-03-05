import unittest
import sys
import numpy as np
# to import from a parent directory
sys.path.append('../')
from NeuralNetwork import NeuralNetwork, mse, rmse, mape
from RNN import RNN
from lstm import LSTM
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

class NeuralNetworksTestCase(unittest.TestCase):
    """ test to ensure each neural network can predict
        a straight line with high accuracy """

    def test_NeuralNetwork(self):
        # create Neural Network
        NN = NeuralNetwork()

        # create training and testing inputs and targets
        train_input = [[100, 100] for i in range(100)]
        train_target = [[100] for i in range(100)]

        test_input = [[101, 101] for i in range(50)]
        test_target = [[101]for i in range(50)]

        # normalize
        train_input = np.array(train_input) / 1000
        train_target = np.array(train_target) / 1000

        test_input = np.array(test_input) / 1000 
        test_target = np.array(test_target)       

        # number of training cycles
        epochs = 100

        # train the neural network
        for e in range(epochs):
            for p in train_input:
                train_output = NN.train(train_input, train_target)

        # test on unseen data
        test_output = NN.test(test_input)

        # de-normalize
        train_output *= 1000
        train_target *= 1000

        test_output *= 1000

        # ensure network can predict a line with high accuracy
        self.assertGreaterEqual(100 - mape(train_target, train_output), 99.99)
        self.assertGreaterEqual(100 - mape(test_target, test_output), 97.00)

    def test_RNN(self):
        # create recurrent neural network
        NN = RNN()

        # create training and testing inputs and targets
        train_input_1 = [[100, 100] for i in range(100)]
        train_target = [[100] for i in range(100)]
        train_input_2 = train_target
        train_input_3 = train_target

        test_input_1 = [[101, 101] for i in range(100)]
        test_target = [[101] for i in range(100)]
        test_input_2 = test_target
        test_input_3 = test_target

        # normalize
        train_input_1 = np.array(train_input_1) / 1000
        train_target = np.array(train_target) / 1000
        train_input_2 = train_target
        train_input_3 = train_target

        test_input_1 = np.array(test_input_1) / 1000
        test_target = np.array(test_target) / 1000
        test_input_2 = test_target
        test_input_3 = test_target        

        # number of training cycles
        epochs = 100

        # train the neural network
        for e in range(epochs):
            for p in train_input_1:
                train_output = NN.train(train_input_1, train_input_2, train_input_3, train_target)

        # test on unseen data
        test_output = NN.test(test_input_1, test_input_2, test_input_3)

        # de-normalize
        train_output *= 1000
        train_target *= 1000

        test_output *= 1000
        test_target *= 1000

        self.assertGreaterEqual(100 - mape(train_target, train_output), 99.99)
        self.assertGreaterEqual(100 - mape(test_target, test_output), 97.00)

    def test_LSTM(self):
        # create recurrent neural network
        NN = LSTM()

        # create training and testing inputs and targets
        train_input_1 = [[100, 100] for i in range(100)]
        train_input_2 = train_input_1
        train_input_3 = train_input_1
        train_target = [[100] for i in range(100)]

        test_input_1 = [[101, 101] for i in range(100)]
        test_input_2 = test_input_1
        test_input_3 = test_input_1
        test_target = [[101] for i in range(100)]

        # normalize
        train_input_1 = np.array(train_input_1) / 1000
        train_input_2 = train_input_1
        train_input_3 = train_input_1
        train_target = np.array(train_target) / 1000

        test_input_1 = np.array(test_input_1) / 1000
        test_input_2 = test_input_1
        test_input_3 = test_input_1       
        test_target = np.array(test_target) / 1000

        # number of training cycles
        epochs = 100

        # train the neural network
        for e in range(epochs):
            for p in train_input_1:
                train_output = NN.train(train_input_1, train_input_2, train_input_3, train_target)

        # test on unseen data
        test_output = NN.test(test_input_1, test_input_2, test_input_3)

        # de-normalize
        train_output *= 1000
        train_target *= 1000

        test_output *= 1000
        test_target *= 1000

        self.assertGreaterEqual(100 - mape(train_target, train_output), 99.99)
        self.assertGreaterEqual(100 - mape(test_target, test_output), 97.00)

if __name__ == '__main__':
    unittest.main()