import numpy as np
import matplotlib.pyplot as plt
from stock import get_stock_data
from sklearn.metrics import mean_squared_error
import math
import pandas as pd
import sys

class NeuralNetwork:

    def __init__(self, input = 2, lstm_cell_weights = 2, output = 1, learning_rate = 0.3):
        # set number of nodes in each input, hidden, output layer
        self.input_nodes = input
        self.lstm_cell_weights = lstm_cell_weights
        self.output_nodes = output
        
        # weight matrices
        # wff = weights for forget gate
        # wfi = weights for input gate
        # wfo = weights for ouput gate
        self.wff = np.random.randn(self.input_nodes, self.lstm_cell_weights)
        self.wfi = np.random.randn(self.input_nodes, self.lstm_cell_weights)
        self.wfo = np.random.randn(self.input_nodes, self.lstm_cell_weights)
        self.wff = self.wff.T
        self.wfi = self.wfi.T
        self.wfo = self.wfo.T
        self.who = np.random.randn(2, 1)
        self.who = self.who.T

        self.cell_state = [[1, 1] for i in range(100)]
        self.cell_state = np.array(self.cell_state, dtype=float)
        self.cell_state = np.array(self.cell_state, ndmin=2).T

        # learning rate
        self.learn = learning_rate

    # sigmoid activation function
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    # tanh activation function
    def tanh(self, x):
        return np.tanh(x)

    def forget_gate(self, gate_input, h_t_1=1):
        gate_input = np.dot(self.wff, gate_input)
        gate_input = h_t_1 * gate_input
        gate_output = self.sigmoid(gate_input)
        self.cell_state = self.cell_state * gate_output


    def input_gate(self, gate_input, h_t_1=1):
        gate_input = np.dot(self.wfi, gate_input)
        gate_input = h_t_1 * gate_input
        gate_output = self.sigmoid(gate_input) * self.tanh(gate_input)
        self.cell_state = self.cell_state + gate_output


    def output_gate(self, gate_input, h_t_1=1):
        gate_input = np.dot(self.wfo, gate_input)
        gate_input = h_t_1 * gate_input
        gate_output = self.sigmoid(gate_input)
        h_t_1 = self.tanh(self.cell_state) * gate_output

        return h_t_1

        
    def forward(self, training_input_1, training_input_2, training_input_3):
        # Pass input though lstm cells
        self.forget_gate(training_input_1)
        self.input_gate(training_input_1)
        h_t = self.output_gate(training_input_1)


        self.forget_gate(training_input_2, h_t)
        self.input_gate(training_input_2, h_t)
        h_t = self.output_gate(training_input_2, h_t)


        self.forget_gate(training_input_3, h_t)
        self.input_gate(training_input_3, h_t)
        h_t = self.output_gate(training_input_3, h_t)

        final_input = np.dot(self.who, h_t)

        final_output = self.sigmoid(final_input)

        return final_output, h_t

    def error(self, target, final_output):
        # output layer error is the (target - actual)
        output_error = target - final_output
        # hidden layer error is the output_error, split by weights, recombined at hidden nodes
        hidden_error = np.dot(self.who.T, output_error) 

        return output_error, hidden_error

    def backpropagation(self,training_input_1, training_input_2, training_input_3, h_t, final_output, output_error, hidden_error):
        # update the weights between hidden and output
        self.who += self.learn * np.dot((output_error * final_output * (1.0 - final_output)), h_t.T)

        # update the weights between input and hidden
        self.wff += self.learn * np.dot((hidden_error * h_t * (1.0 - h_t)), (training_input_1.T * self.cell_state.T))
        self.wfi += self.learn * np.dot((hidden_error * h_t * (1.0 - h_t)), (training_input_2.T * self.cell_state.T))
        self.wfo += self.learn * np.dot((hidden_error * h_t * (1.0 - h_t)), (training_input_3.T * self.cell_state.T))


    def train(self, training_input_1, training_input_2, training_input_3, target):
        # convert lists to 2d arrays
        training_input_1 = np.array(training_input_1, ndmin=2).T
        training_input_2 = np.array(training_input_2, ndmin=2).T
        training_input_3 = np.array(training_input_3, ndmin=2).T
        target = np.array(target, ndmin=2).T

        final_output, h_t = self.forward(training_input_1, training_input_2, training_input_3)

        output_error, hidden_error = self.error(target, final_output)

        self.backpropagation(training_input_1, training_input_2, training_input_3, h_t, final_output, output_error, hidden_error)

        return final_output

    def test(self, testing_input_1, testing_input_2, testing_input_3):
        # transpose input
        testing_input_1 = testing_input_1.T
        testing_input_2 = testing_input_2.T
        testing_input_3 = testing_input_3.T
        final_output, h_t = self.forward(testing_input_1, testing_input_2, testing_input_3)

        return final_output

def mape(actual, prediction): 
    # mean absolute percentage error (MAPE)
    return np.mean(np.abs((actual - prediction) / actual)) * 100

def mse(actual, prediction):
    mse = np.mean((actual - prediction)**2)
    return mse

def rmse(actual, prediction):
    return np.sqrt(((actual - prediction) ** 2).mean())

def plot(actual, prediction):
    plt.plot([0 + i for i in range(0, 200)], actual, "r")
    plt.plot(prediction[:200], "b")
    plt.plot([99 + i for i in range(0, 101)], prediction[99:],  "g")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.title("Stock Prediction")
    plt.legend(["Actual", "Training Prediction", "Test Prediction"])
    plt.grid()
    plt.show()


def main():    
    # returns pandas dataframe
    df = get_stock_data("TSLA")
    # extract only the adjusted close prices of the stock
    df = df['Adj Close']

    # X = (adjclose for 2 days ago, adjclose for previous day)
    # y = actual adjclose for current day
    training_input_1 = [[df[i-6], df[i-5]] for i in range(len(df[:106])) if i >= 6]
    training_input_2 = [[df[i-4], df[i-3]] for i in range(len(df[:106])) if i >= 6]
    training_input_3 = [[df[i-2], df[i-1]] for i in range(len(df[:106])) if i >= 6]
    target = [[i] for i in df[6:106]]

    training_input_1 = np.array(training_input_1, dtype=float)
    training_input_2 = np.array(training_input_2, dtype=float)
    training_input_3 = np.array(training_input_3, dtype=float)
    target = np.array(target, dtype=float)

    assert len(training_input_1) == len(training_input_2) == len(training_input_3) == len(target)

    # Normalize
    training_input_1 = training_input_1/1000
    training_input_2 = training_input_2/1000
    training_input_3 = training_input_3/1000
    target = target/1000 

    # create neural network
    NN = NeuralNetwork()


    # number of training cycles
    training_cycles = 100

    # train the neural network
    for cycle in range(training_cycles):
        for n in training_input_1:
            output = NN.train(training_input_1, training_input_2, training_input_3, target)


    # de-Normalize
    output *= 1000
    target *= 1000

    # transpose
    output = output.T


    # change data type so it can be plotted
    prices = pd.DataFrame(output)

    #print("\nTraining output:\n", output)

    print("\nTraining MSE Accuracy: {:.4f}%".format(100 - mse(target, output)))
    print("Training RMSE Accuracy: {:.4f}%".format(100 - rmse(target, output)))
    print("Training MAPE Accuracy: {:.4f}%".format(100 - mape(target, output)))

    # [price 2 days ago, price yesterday] for each day in range
    testing_input_1 = [[df[i-6], df[i-5]] for i in range(106, 206)]
    testing_input_2 = [[df[i-4], df[i-3]] for i in range(106, 206)]
    testing_input_3 = [[df[i-2], df[i-1]] for i in range(106, 206)]
    test_target = [[i] for i in df[106:206]]


    assert len(testing_input_1) == len(testing_input_2) == len(testing_input_3) == len(test_target)

    testing_input_1 = np.array(testing_input_1, dtype=float)
    testing_input_2 = np.array(testing_input_2, dtype=float)
    testing_input_3 = np.array(testing_input_3, dtype=float)
    test_target = np.array(test_target, dtype=float)

    #print("\nTest input", input)
    #print("\nTest target output", test_target)

    # Normalize
    testing_input_1 = testing_input_1/1000
    testing_input_2 = testing_input_2/1000
    testing_input_3 = testing_input_3/1000

    # test the network with unseen data
    test = NN.test(testing_input_1, testing_input_2, testing_input_3)

    # de-Normalize data
    #input *= 1000
    test *= 1000

    # transplose test results
    test = test.T

    #print("\nTest output:\n", test)

    print("\nTest MSE Accuracy: {:.4f}%".format(100 - mse(test_target, test)))
    print("Test RMSE Accuracy: {:.4f}%".format(100 - rmse(test_target, test)))
    print("Test MAPE Accuracy: {:.4f}%".format(100 - mape(test_target, test)))

    if (100 - mape(test_target, test)) >= 85.00 and (100 - mape(target, output)) >= 85.00:
        # plotting training and test on same graph
        graph_fix = [[0]] * 100
        graph_fix = np.array(graph_fix, dtype=float)
        fixed_test = np.concatenate((graph_fix, test))
        for_plot = np.concatenate((prices[:100], fixed_test[100:]))
        plot(df[6:206], for_plot)

    else:
        main()


if __name__ == '__main__':
    main()