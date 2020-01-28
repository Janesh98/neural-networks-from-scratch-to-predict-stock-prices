import numpy as np
from stock import get_stock_data
import matplotlib.pyplot as plt
from simplenn import Neural_Network, trainer

def mse(a, b):
    return (np.square(np.subtract(a, b)).mean()) / 1000

def main():
    df = get_stock_data("TSLA")
    df = df['Adj Close']

    # X = (adjclose for 2 days ago, adjclose for previous day), y = actual adjclose for current day
    X = [[df[i-1], df[i]] for i in range(len(df[:100])) if i > 1]
    y = [[df[i]] for i in range(len(df)) if i > 2 and i <= 100]

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    assert len(X) == len(y)

    #print(len(X), X, sep="\n")
    #print(len(y), y, sep="\n")

    # Normalize
    X = X/np.amax(X, axis=0)
    y = y/1000 #make y less than 1

    NN = Neural_Network()
    T = trainer(NN)
    T.train(X,y)
    NN.costFunctionPrime(X,y)
    output = NN.forward(X)
    # multiply every element in numpy array by 1000
    output *= 1000
    print(output)

    print("\naccuracy: {:.4f}%".format(mse(y, output)))

    # plot actual price and prediction
    plt.plot(df[:100], "r")
    plt.plot(output, "b")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.title("Stock Prediction")
    plt.legend(["Actual", "Prediction"])
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
