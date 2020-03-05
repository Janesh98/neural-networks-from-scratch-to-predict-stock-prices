from FeedForward import FeedForward
import matplotlib.pyplot as plt
from stock import get_stock_data, plot
import pandas as pd
from normalize import Normalize
import numpy as np
from utils import *

def ff_main():    
    # returns pandas dataframe
    df = get_stock_data("TSLA")
    # extract only the adjusted close prices of the stock
    df = df['Adj Close']

    # X = (adjclose for 2 days ago, adjclose for previous day)
    # y = actual adjclose for current day
    X = [[df[i-2], df[i-1]] for i in range(len(df[:102])) if i >= 2]
    y = [[i] for i in df[2:102]]

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    assert len(X) == len(y)
    
    # Normalize
    scaler = Normalize(df)
    X = scaler.normalize_data(X)
    y = scaler.normalize_data(y)

    # create neural network
    NN = FeedForward()

    # number of training cycles
    epochs = 100

    # train the neural network
    for e in range(epochs):
        for n in X:
            output = NN.train(X, y)

    # de-Normalize
    output = scaler.denormalize_data(output)
    y = scaler.denormalize_data(y)

    # transpose
    output = output.T

    # change data type so it can be plotted
    prices = pd.DataFrame(output)

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
    input = scaler.normalize_data(input)

    # test the network with unseen data
    test = NN.test(input)

    # de-Normalize data
    input = scaler.denormalize_data(input)
    test = scaler.denormalize_data(test)

    # transplose test results
    test = test.T

    #print("\nTest output:\n", test)

    print("\nTest MSE Accuracy: {:.4f}%".format(100 - mse(test_target, test)))
    print("Test RMSE Accuracy: {:.4f}%".format(100 - rmse(test_target, test)))
    print("Test MAPE Accuracy: {:.4f}%".format(100 - mape(test_target, test)))

    plot(df[2:152], output, test)

if __name__ == "__main__":
    ff_main()