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
        # wih = weights from input(i) to hidden(h)
        # who = weights from hidden(i) to output(o)
        self.wih1 = np.random.randn(self.input_nodes, self.hidden_nodes_1)
        self.wih1 = self.wih1.T
        self.wh1h2 = np.random.randn(self.hidden_nodes_1, self.hidden_nodes_2)
        self.wh2o = np.random.randn(self.hidden_nodes_2, self.output_nodes)
        self.wh2o = self.wh2o.T

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

    def forward(self, training_input_1, training_input_2, training_input_3):
        # calculate signals into hidden layer 1
        hidden_input = np.dot(self.wih1, training_input_1)
        # calculate the signals emerging from hidden layer 1
        hidden_output_1 = self.tanh(hidden_input)

        # add new input for next hidden layer
        hidden_output_1 = np.insert(hidden_output_1, 0, training_input_2, axis=0)
        # calculate signals into hidden layer 2
        hidden_input_2 = np.dot(self.wh1h2, hidden_output_1)
        # calculate the signals emerging from hidden layer 1
        hidden_output_2 = self.tanh(hidden_input_2)

        # add new input for final output
        hidden_output_2 = np.insert(hidden_output_2, 0, training_input_3, axis=0)
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
        # update the weights between hidden and output
        self.wh2o += self.learn * np.dot((output_error * final_output * (1.0 - final_output)), hidden_output_2.T)
        self.wh1h2 += self.learn * np.dot((output_error * final_output * (1.0 - final_output)), hidden_output_1.T)
        
        # update the weights between input and hidden
        self.wih1 += self.learn * np.dot((hidden_error_2[1:] * hidden_output_1[1:] * (1.0 - hidden_output_1[1:])), training_input_1.T)
        
        
    def train(self, training_input_1, training_input_2, training_input_3, target):
        # convert lists to 2d arrays
        training_input_1 = np.array(training_input_1, ndmin=2).T
        training_input_2 = np.array(training_input_2, ndmin=2).T
        training_input_3 = np.array(training_input_3, ndmin=2).T
        target = np.array(target, ndmin=2).T

        final_output, hidden_output_1, hidden_output_2 = self.forward(training_input_1, training_input_2, training_input_3)

        output_error, hidden_error_2 = self.error(target, final_output)

        self.backpropagation(training_input_1, hidden_output_1, hidden_output_2, final_output, output_error, hidden_error_2)

        return final_output

    def test(self, testing_input_1, testing_input_2, testing_input_3):
        # transpose input
        testing_input_1 = testing_input_1.T
        testing_input_2 = testing_input_2.T
        testing_input_3 = testing_input_3.T
        final_output, hidden_output_1, hidden_output_2 = self.forward(testing_input_1, testing_input_2, testing_input_3)

        return final_output

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

def plot(actual, prediction):
    plt.plot([0 + i for i in range(0, 150)], actual, "r")
    plt.plot(prediction[:100], "b")
    plt.plot([99 + i for i in range(0, 51)], prediction[99:],  "g")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.title("Stock Prediction")
    plt.legend(["Actual", "Training Prediction", "Test Prediction"])
    plt.grid()
    plt.show()

def mape(actual, prediction): 
    # mean absolute percentage error (MAPE)
    return np.mean(np.abs((actual - prediction) / actual)) * 100

def mse(actual, prediction):
    mse = np.mean((actual - prediction)**2)
    return mse

def rmse(actual, prediction):
    return np.sqrt(((actual - prediction) ** 2).mean())

def main():    
    # returns pandas dataframe
    df = get_stock_data("TSLA")
    # extract only the adjusted close prices of the stock
    df = df['Adj Close']

    # X = (adjclose for 2 days ago, adjclose for previous day)
    # y = actual adjclose for current day
    training_input_1 = [[df[i-4], df[i-3]] for i in range(len(df[:104])) if i >= 4]
    training_input_2 = [[df[i-2]] for i in range(len(df[:104])) if i >= 4] 
    training_input_3 = [[df[i - 1]] for i in range(len(df[:104])) if i >= 4]
    target = [[i] for i in df[4:104]]

    training_input_1 = np.array(training_input_1, dtype=float)
    training_input_2 = np.array(training_input_2, dtype=float)
    training_input_3 = np.array(training_input_3, dtype=float)
    target = np.array(target, dtype=float)


    assert len(training_input_1) == len(training_input_2) == len(training_input_3) == len(target)

    #print("\ninput:\n", X)
    #print("\nTraining target output:", y)

    # Normalize
    training_input_1 = training_input_1/1000
    training_input_2 = training_input_2/1000
    training_input_3 = training_input_3/1000
    target = target/1000 #make y less than 1

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
    testing_input_1 = [[df[i-4], df[i-3]] for i in range(104, 154)]
    testing_input_2 = [[df[i-2]] for i in range(104, 154)] 
    testing_input_3 = [[df[i-1]] for i in range(104, 154)]
    test_target = [[i] for i in df[104:154]]


    assert len(testing_input_1) == len(testing_input_2) == len(testing_input_3) ==len(test_target)

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

    if (100 - mape(test_target, test)) >= 80.00:
        # plotting training and test on same graph
        graph_fix = [[0]] * 100
        graph_fix = np.array(graph_fix, dtype=float)
        fixed_test = np.concatenate((graph_fix, test))
        for_plot = np.concatenate((prices[:100], fixed_test[100:]))
        plot(df[2:152], for_plot)

    else:
        main()


if __name__ == "__main__":
    main()