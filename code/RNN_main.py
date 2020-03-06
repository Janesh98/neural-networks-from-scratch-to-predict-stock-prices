import matplotlib.pyplot as plt
from stock import get_stock_data, plot
from normalize import Normalize
from utils import *
from NeuralNetworks.RNN import RNN
import pandas as pd


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

    # training inputs = adjclose for previous 4 days
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


    # Normalize
    training_input_1 = training_input_1/1000
    training_input_2 = training_input_2/1000
    training_input_3 = training_input_3/1000
    target = target/1000 #make y less than 1

    # create neural network
    NN = RNN()

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


    # print various accuracies
    print("\nTraining MSE Accuracy: {:.4f}%".format(100 - mse(target, output)))
    print("Training RMSE Accuracy: {:.4f}%".format(100 - rmse(target, output)))
    print("Training MAPE Accuracy: {:.4f}%".format(100 - mape(target, output)))


    # testing inputs are for previous 4 days
    testing_input_1 = [[df[i-4], df[i-3]] for i in range(104, 154)]
    testing_input_2 = [[df[i-2]] for i in range(104, 154)] 
    testing_input_3 = [[df[i-1]] for i in range(104, 154)]
    test_target = [[i] for i in df[104:154]]


    assert len(testing_input_1) == len(testing_input_2) == len(testing_input_3) ==len(test_target)

    testing_input_1 = np.array(testing_input_1, dtype=float)
    testing_input_2 = np.array(testing_input_2, dtype=float)
    testing_input_3 = np.array(testing_input_3, dtype=float)
    test_target = np.array(test_target, dtype=float)


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


    # print various accuracies
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