import numpy as np
import matplotlib.pyplot as plt
from stock import get_stock_data
from sklearn.metrics import mean_squared_error
import math
import pandas as pd
import sys

class NeuralNetwork:

    def __init__(self, input = 2, hidden = 3, output = 1, learning_rate = 0.3):
        # set number of nodes in each input, hidden, output layer
        self.input_nodes = input
        self.hidden_nodes = hidden
        self.output_nodes = output
        
        # weight matrices
        # wih = weights from input(i) to hidden(h)
        # who = weights from hidden(i) to output(o)
        self.wih = np.random.randn(self.input_nodes, self.hidden_nodes)
        self.wih = self.wih.T
        self.who = np.random.randn(self.hidden_nodes, self.output_nodes)
        self.who = self.who.T

        # learning rate
        self.learn = learning_rate

    # sigmoid activation function
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def forward(self, input):
        # calculate signals into hidden layer
        hidden_input = np.dot(self.wih, input)
        # calculate the signals emerging from hidden layer
        hidden_output = self.sigmoid(hidden_input)
        
        # calculate signals into final output layer
        final_input = np.dot(self.who, hidden_output)

        # calculate the signals emerging from final output layer
        final_output = self.sigmoid(final_input)

        return final_output, hidden_output

    def error(self, target, final_output):
        # output layer error is the (target - actual)
        output_error = target - final_output
        # hidden layer error is the output_error, split by weights, recombined at hidden nodes
        hidden_error = np.dot(self.who.T, output_error) 

        return output_error, hidden_error

    def backpropagation(self, input, hidden_output, final_output, output_error, hidden_error):
        # update the weights between hidden and output
        self.who += self.learn * np.dot((output_error * final_output * (1.0 - final_output)), hidden_output.T)
        
        # update the weights between input and hidden
        self.wih += self.learn * np.dot((hidden_error * hidden_output * (1.0 - hidden_output)), input.T)
        
        
    def train(self, input, target):
        # convert lists to 2d arrays
        input = np.array(input, ndmin=2).T
        target = np.array(target, ndmin=2).T

        final_output, hidden_output = self.forward(input)

        output_error, hidden_error = self.error(target, final_output)

        self.backpropagation(input, hidden_output, final_output, output_error, hidden_error)

        return final_output

    def test(self, input):
        # transpose input
        input = input.T
        final_output, hidden_output = self.forward(input)

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


def mape(actual, prediction): 
    # mean absolute percentage error (MAPE)
    return np.mean(np.abs((actual - prediction) / actual)) * 100

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

    print(df)

    # X = (adjclose for 2 days ago, adjclose for previous day)
    # y = actual adjclose for current day
    X = [[df[i-2], df[i-1]] for i in range(len(df[:102])) if i >= 2]
    y = [[i] for i in df[2:102]]

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)


    assert len(X) == len(y)
    print(len(X), len(y))

    print("\ninput:\n", X)
    print("\nTraining target output:", y)

    # Normalize
    X = X/1000
    y = y/1000 #make y less than 1

    input_nodes = 2
    hidden_nodes = 3
    output_nodes = 1

    learning_rate = 0.3

    # create neural network
    NN = NeuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)

    # number of training cycles
    epochs = 100

    # train the neural network
    for e in range(epochs):
        for n in X:
            output = NN.train(X, y)

    # de-Normalize
    output *= 1000
    y *= 1000

    # transpose
    output = output.T

    # change data type so it can be plotted
    prices = pd.DataFrame(output)

    #print("\nTraining output:\n", output)

    print("\nTraining MSE Accuracy: {:.4f}%".format(100 - mse(y, output)))
    print("Training RMSE Accuracy: {:.4f}%".format(100 - rmse(y, output)))
    print("Training MAPE Accuracy: {:.4f}%".format(100 - mape(y, output)))

    # [price 2 days ago, price yesterday] for each day in range
    input = [[df[i-2], df[i-1]] for i in range(102, 152)]
    test_target = [[i] for i in df[102:152]]

    assert len(input) == len(test_target)

    input = np.array(input, dtype=float)
    test_target = np.array(test_target, dtype=float)

    #print("\nTest input", input)
    #print("\nTest target output", test_target)

    # Normalize
    input = input/1000

    # test the network with unseen data
    test = NN.test(input)

    # de-Normalize data
    input *= 1000
    test *= 1000

    # transplose test results
    test = test.T

    #print("\nTest output:\n", test)

    print("\nTest MSE Accuracy: {:.4f}%".format(100 - mse(test_target, test)))
    print("Test RMSE Accuracy: {:.4f}%".format(100 - rmse(test_target, test)))
    print("Test MAPE Accuracy: {:.4f}%".format(100 - mape(test_target, test)))

    # plotting training and test on same graph
    graph_fix = [[0]] * 100
    graph_fix = np.array(graph_fix, dtype=float)
    fixed_test = np.concatenate((graph_fix, test))
    for_plot = np.concatenate((prices[:100], fixed_test[100:]))

    plot(df[2:152], for_plot)

if __name__ == "__main__":
    main()