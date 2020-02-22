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

def train_test_split(df, split=0.75):
    # if split=0.75, splits data into 75% training, 25% test
    # provides targets for training and accuracy measurments
    max_index = round((len(df) - 1) * split)

    # adjusted close price [2 days ago, 1 day ago]
    train_inputs = [[df[i-2], df[i-1]] for i in range(2, max_index)]
    # target is the next day for a given input above
    # e.g inputs = [day1, day2], [day2, day3]
    #     targets = [day3, day4]
    train_targets = [[i] for i in df[2 : max_index]]

    assert len(train_inputs) == len(train_targets)

    test_inputs = [[df[i-2], df[i-1]] for i in range(max_index + 2, len(df))]
    test_targets = [[i] for i in df[max_index + 2:]]

    assert len(test_inputs) == len(test_targets)

    return train_inputs, train_targets, test_inputs, test_targets

def predict(stock, start, end):
    input_nodes = 2
    hidden_nodes = 3
    output_nodes = 1
    learning_rate = 0.3

    # create neural network
    NN = NeuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)

    # get stock data
    df = get_stock_data(stock, start, end, json=False)

    # split data into training and testing
    train_inputs, train_targets, test_inputs, test_targets = train_test_split(df)

    # normalize data
    normalising_factor = NN.normalise_factor(train_inputs)
    train_inputs = NN.normalise_data(train_inputs, normalising_factor)
    train_targets = NN.normalise_data(train_targets, normalising_factor)

    # number of training cycles
    epochs = 100

    # train the neural network
    for epoch in range(epochs):
        for prices in train_inputs:
            train_outputs = NN.train(train_inputs, train_targets)

    # de-Normalize data
    train_outputs = NN.denormalise_data(train_outputs, normalising_factor)
    prices = pd.DataFrame(train_outputs.T)

    # Normalize data
    test_inputs = NN.normalise_data(test_inputs, normalising_factor)

    # test the network with unseen data
    test_outputs = NN.test(test_inputs)

    # de-Normalize data
    test_inputs = NN.denormalise_data(test_inputs, normalising_factor)
    test_outputs = NN.denormalise_data(test_outputs, normalising_factor)

    # transplose test results
    test_outputs = test_outputs.T

    # return original stock data, training output, testing output
    return df, prices, pd.DataFrame(test_outputs)


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