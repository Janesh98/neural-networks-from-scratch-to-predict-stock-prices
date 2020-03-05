import matplotlib.pyplot as plt
from stock import get_stock_data, plot
from normalize import Normalize
from utils import *
from NeuralNetworks.rnn_v2 import RNN_V2

def rnn_main():
    df = get_stock_data("TSLA")
    stock = df["Adj Close"].values

    # normalize
    scaler = Normalize(stock)
    normalized = scaler.normalize_data(stock)

    # get training and testing inputs and outputs
    train_inputs, train_targets, test_inputs, test_targets = train_test_split(normalized)

    # returns 3d array in format [inputs, timesteps, features]
    train_inputs = to_3d(train_inputs)
    test_inputs = to_3d(test_inputs)

    #print(train_inputs.shape, train_targets.shape)
    #print(test_inputs.shape, test_targets.shape)

    NN = RNN_V2()
    train_outputs = NN.train(train_inputs, train_targets, epochs=100)
    test_outputs = NN.test(test_inputs, test_targets)

    # de-normalize
    train_outputs = scaler.denormalize_data(train_outputs)
    train_targets = scaler.denormalize_data(train_targets)
    test_outputs = scaler.denormalize_data(test_outputs)
    test_targets = scaler.denormalize_data(test_targets)

    print("\nTraining MSE Accuracy: {:.4f}%".format(100 - mse(train_targets, train_outputs)))
    print("Training RMSE Accuracy: {:.4f}%".format(100 - rmse(train_targets, train_outputs)))
    print("Training MAPE Accuracy: {:.4f}%".format(100 - mape(train_targets, train_outputs)))

    print("\nTest MSE Accuracy: {:.4f}%".format(100 - mse(test_targets, test_outputs)))
    print("Test RMSE Accuracy: {:.4f}%".format(100 - rmse(test_targets, test_outputs)))
    print("Test MAPE Accuracy: {:.4f}%".format(100 - mape(test_targets, test_outputs)))
    
    # plot the results compared to the original stock data
    plot(stock, train_outputs, test_outputs)

if __name__ == "__main__":
    rnn_main()