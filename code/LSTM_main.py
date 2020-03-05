from NeuralNetworks.lstm import LSTM
import numpy as np
from stock import get_stock_data, plot
import pandas as pd

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
    df = get_stock_data("FB")
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
    NN = LSTM()

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
    testing_input_1 = [[df[i-6], df[i-5]] for i in range(106, 156)]
    testing_input_2 = [[df[i-4], df[i-3]] for i in range(106, 156)]
    testing_input_3 = [[df[i-2], df[i-1]] for i in range(106, 156)]
    test_target = [[i] for i in df[106:156]]

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
    test_target = test_target/1000

    # test the network with unseen data
    test = NN.test(testing_input_1, testing_input_2, testing_input_3, test_target)

    # de-Normalize data
    test *= 1000
    test_target *= 1000

    # transplose test results
    test = test.T


    #print("\nTest output:\n", test)

    print("\nTest MSE Accuracy: {:.4f}%".format(100 - mse(test_target, test)))
    print("Test RMSE Accuracy: {:.4f}%".format(100 - rmse(test_target, test)))
    print("Test MAPE Accuracy: {:.4f}%".format(100 - mape(test_target, test)))

    # plotting training and test on same graph

    plot(df, output, test)

if __name__ == '__main__':
    main()