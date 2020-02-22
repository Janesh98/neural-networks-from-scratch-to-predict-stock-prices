from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pandas_datareader.data as web
import datetime as dt
from os import listdir
import sys
# to import from a parent directory
sys.path.append('../')
from NeuralNetwork import NeuralNetwork

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
# prevent caching so website can be updated dynamically
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

def predict(stock, start, end):
    input_nodes = 2
    hidden_nodes = 3
    output_nodes = 1

    learning_rate = 0.3

    # create neural network
    NN = NeuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)

    df = get_stock_data(stock, start, end, json=False)

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

    return df, prices, pd.DataFrame(test)


def get_stock_data(ticker, start=[2019, 1, 1], end=[2019, 12, 31], json=True):
    # *list passes the values in list as parameters
    start = dt.datetime(*start)
    end = dt.datetime(*end)

    # download csv from yahoo finance
    df = web.DataReader(ticker, 'yahoo', start, end)
    # extract adjusted close column
    df = df["Adj Close"]
    # remove Date column
    df = pd.DataFrame([i for i in df])[0]
    if json:
        # return data as JSON
        return df.to_json()

    else:
        # return data as csv
        return df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/getpythondata')
def get_python_data():
    return get_stock_data("TSLA")

@app.route('/hello', methods=['POST'])
def hello():
    # POST request
    if request.method == 'POST':
        print('Incoming..')
        # convert to JSON
        data = request.get_json(force=True)
        stock = data["stock"].upper()
        start = data["startDate"].split("-")
        end = data["endDate"].split("-")

        # convert strings to integers
        start, end = [int(s) for s in start], [int(s) for s in end]
        print(stock, start, end)

        actual, train_res, test_res = predict(stock, start, end)

        # convert pandas dataframe to list
        actual = [i for i in actual]
        train_res, test_res = [i for i in train_res[0][:]], [i for i in test_res[0][:]]

        # range of y values for plotting
        actualX = [i for i in range(len(actual))] 
        trainX = [i for i in range(len(train_res))] 
        testX = [i for i in range(len(train_res) - 1, len(train_res) + len(test_res))]

        return {"stock" : stock,
                "actual" : actual,
                "actualX" : actualX,
                "train" : train_res,
                "trainX" : trainX, 
                "test" : test_res,
                "testX" : testX}
        
if __name__ == '__main__':
    app.run(host = "0.0.0.0", port = 3000, debug=True)