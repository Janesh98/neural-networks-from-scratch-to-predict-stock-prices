import numpy as np
import matplotlib.pyplot as plt
from stock import get_stock_data
import pandas as pd
import sys

class NeuralNetwork:

    def __init__(self, input, hidden, output, learning_rate):
        # set number of nodes in each input, hidden, output layer
        self.input_nodes = input
        self.hidden_nodes = hidden
        self.output_nodes = output
        
        # weight matrices
        # wih = weights from input(i) to hidden(h)
        self.wih = np.random.normal(0.0, pow(self.input_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
        self.who = np.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.output_nodes, self.hidden_nodes))

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

def main():    
    # returns pandas dataframe
    df = get_stock_data("TSLA")
    # extract only the adjusted close prices of the stock
    df = df['Adj Close']

    # X = (adjclose for 2 days ago, adjclose for previous day)
    # y = actual adjclose for current day
    X = [[df[i-1], df[i]] for i in range(len(df[:101])) if i >= 1]
    y = [[df[i]] for i in range(len(df)) if i > 1 and i <= 101]

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    assert len(X) == len(y)

    print("\ninput:\n", X)
    print("\ntarget output:", y)

    # Normalize
    X = X/np.amax(X, axis=0)
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

    # change data type so it can be plotted
    prices = pd.DataFrame(output.T)
    print("\nTraining output:\n", prices)
    print("\nTraining Accuracy: {:.4f}%".format(mape(output, y)))


    # [price yesterday, current price] for each day in range
    input = [[df[i-1], df[i]] for i in range(100, 150)]

    # Normalize data
    input = np.array(input, dtype=float)
    input = input/1000

    # test the network with unseen data
    test = NN.test(input)

    # de-Normalize data
    input *= 1000
    test *= 1000

    # transplose test results
    test = test.T

    print("\nTest output:\n", test)
    print("\nTest Accuracy: {:.4f}%".format(mape(input, test)))


    # plot actual price and prediction for training and test
    # TODO plot in same window but seperate graphs
    # plot(df[:100], prices)
    # plot(df[:len(input)], test)

    # plotting training and test on same graph
    graph_fix = [[0]] * 100
    graph_fix = np.array(graph_fix, dtype=float)
    fixed_test = np.concatenate((graph_fix, test))
    for_plot = np.concatenate((prices[:100], fixed_test[100:]))
    plot(df[2:152], for_plot)


if __name__ == "__main__":
    main()