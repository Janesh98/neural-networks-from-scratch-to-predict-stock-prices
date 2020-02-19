import numpy as np
import matplotlib.pyplot as plt
from stock import get_stock_data
import pandas as pd
import sys
from NeuralNetwork import NeuralNetwork

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
    input_nodes = 2
    hidden_nodes = 3
    output_nodes = 1

    learning_rate = 0.3

    # create neural network
    NN = NeuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)

    ticker = input("Enter company's ticker symbol(e.g. TSLA)\n")
    ticker = ticker.upper()
    df = get_stock_data(ticker)
    df = df['Adj Close']


    # X = (adjclose for 2 days ago, adjclose for previous day)
    # y = actual adjclose for current day
    X = [[df[i-1], df[i]] for i in range(len(df[:101])) if i >= 1]
    y = [[df[i]] for i in range(len(df)) if i > 1 and i <= 101]

    normalising_factor = NN.normalise_factor(X)

    X = NN.normalise_data(X, normalising_factor)
    y = NN.normalise_data(y, normalising_factor)

    assert len(X) == len(y)

    # number of training cycles
    training_cycles = 100

    # train the neural network
    for cyclewi in range(training_cycles):
        for n in X:
            output = NN.train(X, y)

    output = NN.denormalise_data(output, normalising_factor)
    prices = pd.DataFrame(output.T)

    # [price yesterday, current price] for each day in range
    inputs = [[df[i-1], df[i]] for i in range(100, 150)]

    # Normalize data
    inputs = NN.normalise_data(inputs, normalising_factor)

    # test the network with unseen data
    test = NN.test(inputs)

    # de-Normalize data
    inputs = NN.denormalise_data(inputs, normalising_factor)
    test = NN.denormalise_data(test, normalising_factor)

    # transplose test results
    test = test.T

    graph_fix = [[0]] * 100
    graph_fix = np.array(graph_fix, dtype=float)
    fixed_test = np.concatenate((graph_fix, test))
    for_plot = np.concatenate((prices[:100], fixed_test[100:]))
    plot(df[2:152], for_plot)


if __name__ == '__main__':
    main()