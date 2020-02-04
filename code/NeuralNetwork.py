import numpy as np
import matplotlib.pyplot as plt
from stock import get_stock_data
import pandas as pd

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
        
    def train(self, input, target):
        # convert inputs and target list to 2d arrays
        input = np.array(input, ndmin=2).T
        target = np.array(target, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_input = np.dot(self.wih, input)
        # calculate the signals emerging from hidden layer
        hidden_output = self.sigmoid(hidden_input)
        
        # calculate signals into final output layer
        final_input = np.dot(self.who, hidden_output)
        # calculate the signals emerging from final output layer
        final_output = self.sigmoid(final_input)
        
        # output layer error is the (target - actual)
        output_error = target - final_output
        # hidden layer error is the output_error, split by weights, recombined at hidden nodes
        hidden_error = np.dot(self.who.T, output_error) 
        
        # update the weights for the links between the hidden and output layers
        self.who += self.learn * np.dot((output_error * final_output * (1.0 - final_output)), np.transpose(hidden_output))
        
        # update the weights for the links between the input and hidden layers
        self.wih += self.learn * np.dot((hidden_error * hidden_output * (1.0 - hidden_output)), np.transpose(input))

        return final_output

    def test(self, input):
        # ranspose input
        input = input.T
        
        # calculate signals into hidden layer
        hidden_input = np.dot(self.wih, input)
        # calculate the signals emerging from hidden layer
        hidden_output = self.sigmoid(hidden_input)
        
        # calculate signals into final output layer
        final_input = np.dot(self.who, hidden_output)

        # calculate the signals emerging from final output layer
        final_output = self.sigmoid(final_input)

        return final_output

def mse(a, b):
    return (np.square(np.subtract(a, b)).mean()) / 1000

def plot(actual, prediction):
    plt.plot(actual, "r")
    plt.plot(prediction, "b")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.title("Stock Prediction")
    plt.legend(["Actual", "Prediction"])
    plt.grid()
    plt.show()

def main():    
    # returns pandas dataframe
    df = get_stock_data("TSLA")
    # extract only the adjusted close prices of the stock
    df = df['Adj Close']

    # X = (adjclose for 2 days ago, adjclose for previous day)
    # y = actual adjclose for current day
    X = [[df[i-1], df[i]] for i in range(len(df[:100])) if i > 1]
    y = [[df[i]] for i in range(len(df)) if i > 2 and i <= 100]

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
    print("\nTraining Accuracy: {:.4f}%".format(mse(output, y)))

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
    print("\nTest Accuracy: {:.4f}%".format(mse(input, test)))

    # plot actual price and prediction for training and test
    # TODO plot in same window but seperate graphs
    plot(df[:100], prices)
    plot(df[:len(input)], test)

if __name__ == "__main__":
    main()