import unittest
import numpy as np
import sys
# to import from a parent directory
sys.path.append('../')
from utils import mape, to_3d
from NeuralNetworks.FeedForward import FeedForward
from NeuralNetworks.RNN import RNN
from NeuralNetworks.rnn_v2 import RNN_V2
from NeuralNetworks.lstm import LSTM

class NeuralNetworksTestCase(unittest.TestCase):
    """ test to ensure each neural network can predict
        a straight line with high accuracy """

    def test_FeedForward(self):
        # create Neural Network
        NN = FeedForward()

        # create training and testing inputs and targets
        train_input = [[100, 100] for i in range(100)]
        train_target = [[100] for i in range(100)]

        test_input = [[101, 101] for i in range(50)]
        test_target = [[101]for i in range(50)]

        # normalize
        train_input = np.array(train_input) / 1000
        train_target = np.array(train_target) / 1000
        test_input = np.array(test_input) / 1000 

        # convert to array
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
        self.assertGreaterEqual(100 - mape(train_target, train_output), 99.00)
        self.assertGreaterEqual(100 - mape(test_target, test_output), 97.00)

    def test_RNN(self):
        # create recurrent neural network
        NN = RNN()

        # create training and testing inputs and targets
        train_input_1 = [[100, 100] for i in range(100)]
        train_target = [[100] for i in range(100)]
        train_input_2 = train_target
        train_input_3 = train_target

        test_input_1 = [[101, 101] for i in range(50)]
        test_target = [[101] for i in range(50)]
        test_input_2 = test_target
        test_input_3 = test_target

        # convert to array and normalize
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

        self.assertGreaterEqual(100 - mape(train_target, train_output), 99.00)
        self.assertGreaterEqual(100 - mape(test_target, test_output), 97.00)

    def test_RNN_V2(self):
        # create Neural Network
        NN = RNN_V2()

        # create training and testing inputs and targets
        train_input = [[100, 100] for i in range(100)]
        train_target = [[100] for i in range(100)]

        test_input = [[101, 101] for i in range(50)]
        test_target = [[101]for i in range(50)]

        # normalize
        train_input = np.array(train_input) / 1000
        train_target = np.array(train_target) / 1000
        test_input = np.array(test_input) / 1000 

        # convert to array
        test_target = np.array(test_target)

        # convert to 3d array of format [inputs, timesteps, features]
        train_input = to_3d(train_input)
        test_input = to_3d(test_input)      

        # train the neural network
        train_output = NN.train(train_input, train_target, epochs=100)

        # test on unseen data
        test_output = NN.test(test_input)

        # de-normalize
        train_output *= 1000
        train_target *= 1000

        test_output *= 1000

        # ensure network can predict a line with high accuracy
        self.assertGreaterEqual(100 - mape(train_target, train_output), 99.00)
        self.assertGreaterEqual(100 - mape(test_target, test_output), 97.00)

    def test_LSTM(self):
        # create recurrent neural network
        NN = LSTM()

        # create training and testing inputs and targets
        train_input_1 = [[100, 100] for i in range(100)]
        train_input_2 = train_input_1
        train_input_3 = train_input_1
        train_target = [[100] for i in range(100)]

        test_input_1 = [[101, 101] for i in range(50)]
        test_input_2 = test_input_1
        test_input_3 = test_input_1
        test_target = [[101] for i in range(50)]

        # normalize and convert to arrays
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

        self.assertGreaterEqual(100 - mape(train_target, train_output), 99.00)
        self.assertGreaterEqual(100 - mape(test_target, test_output), 97.00)


if __name__ == '__main__':
    unittest.main()