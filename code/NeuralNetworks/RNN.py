import numpy as np
import matplotlib.pyplot as plt
from stock import get_stock_data
from sklearn.metrics import mean_squared_error
import math
import pandas as pd
import sys

class RNN:

    def __init__(self, input_1 = 2, input_2 = 1, input_3 = 1, hidden_layer_1 = 2, hidden_layer_2 = 3, output = 1, learning_rate = 0.3):
        # set number of nodes in each input, hidden, output layer
        self.input_nodes = input_1
        self.input_2 = input_2
        self.input_2 = input_3
        self.hidden_nodes_1 = hidden_layer_1
        self.hidden_nodes_2 = hidden_layer_2
        self.output_nodes = output
        
        # weight matrices
        # wih1 = weights from input(i) to hidden(h) layer 1
        # wh1h2 = weights from hidden(h) layer 1 to hidden(h) layer 2
        # wh2o = weights from hidden(h) layer 2 to output(o) 
        self.wih1 = np.random.randn(self.input_nodes, self.hidden_nodes_1).T
        self.wh1h2 = np.random.randn(self.hidden_nodes_1, self.hidden_nodes_2)
        self.wh2o = np.random.randn(self.hidden_nodes_2, self.output_nodes).T

        # learning rate
        self.learn = learning_rate

    # sigmoid activation function
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    # tanh activation function
    def tanh(self, x):
        return np.tanh(x)
    
    # relu activation function
    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, input_1, input_2, input_3):
        # calculate signals into hidden layer 1
        hidden_input = np.dot(self.wih1, input_1)
        # calculate the signals emerging from hidden layer 1
        hidden_output_1 = self.tanh(hidden_input)

        # add new input for next hidden layer
        hidden_output_1 = np.insert(hidden_output_1, 0, input_2, axis=0)
        # calculate signals into hidden layer 2
        hidden_input_2 = np.dot(self.wh1h2, hidden_output_1)
        # calculate the signals emerging from hidden layer 1
        hidden_output_2 = self.tanh(hidden_input_2)

        # add new input for final output
        hidden_output_2 = np.insert(hidden_output_2, 0, input_3, axis=0)
        # calculate signals into final output layer
        final_input = np.dot(self.wh2o, hidden_output_2)

        # calculate the signals emerging from final output layer
        final_output = self.sigmoid(final_input)

        return final_output, hidden_output_1, hidden_output_2

    def error(self, target, final_output):
        # output layer error is the (target - actual)
        output_error = target - final_output
        # hidden layer error is the output_error, split by weights, recombined at hidden nodes
        hidden_error_2 = np.dot(self.wh2o.T, output_error)  

        return output_error, hidden_error_2

    def backpropagation(self, training_input_1, hidden_output_1, hidden_output_2, final_output, output_error, hidden_error_2):
        # update the weights between hidden layers and output
        self.wh2o += self.learn * np.dot((output_error * final_output * (1.0 - final_output)), hidden_output_2.T)
        self.wh1h2 += self.learn * np.dot((output_error * final_output * (1.0 - final_output)), hidden_output_1.T)
        
        # update the weights between input and hidden layer 1
        self.wih1 += self.learn * np.dot((hidden_error_2[1:] * hidden_output_1[1:] * (1.0 - hidden_output_1[1:])), training_input_1.T)

        # clip to prevent/reduce exploding/vanishing gradient problem
        for w in [self.wh2o, self.wh1h2, self.wih1]:
            np.clip(w, -5, 5, out=w)
        
        
    def train(self, training_input_1, training_input_2, training_input_3, target):
        # convert lists to 2d arrays
        training_input_1 = np.array(training_input_1, ndmin=2).T
        training_input_2 = np.array(training_input_2, ndmin=2).T
        training_input_3 = np.array(training_input_3, ndmin=2).T
        target = np.array(target, ndmin=2).T

        # forward propogation to return final output and hidden layers output
        final_output, hidden_output_1, hidden_output_2 = self.forward(training_input_1, training_input_2, training_input_3)

        # calculate errors
        output_error, hidden_error_2 = self.error(target, final_output)

        self.backpropagation(training_input_1, hidden_output_1, hidden_output_2, final_output, output_error, hidden_error_2)

        return final_output

    def test(self, testing_input_1, testing_input_2, testing_input_3):
        # transpose input
        testing_input_1 = testing_input_1.T
        testing_input_2 = testing_input_2.T
        testing_input_3 = testing_input_3.T
        final_output, hidden_output_1, hidden_output_2 = self.forward(testing_input_1, testing_input_2, testing_input_3)
        # return final prediction
        return final_output

    #calculate normalising factor
    def normalise_factor(self, data):
        biggest = max(data)
        if len(biggest) > 1:
            biggest = max(biggest)
        else:
            biggest = biggest[0]

        head = int(biggest)
        return 10 ** len(str(head))

    def normalise_data(self, data, factor):
        data = np.array(data, dtype=float)
        return data/factor

    def denormalise_data(self, data, factor):
        return data * factor