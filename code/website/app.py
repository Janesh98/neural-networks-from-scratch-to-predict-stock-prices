from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pandas_datareader.data as web
import datetime as dt
import sys
# to import from a parent directory
sys.path.append('../')
from NeuralNetworks.FeedForward import FeedForward
from NeuralNetworks.rnn_v2 import RNN_V2
from NeuralNetworks.lstm import LSTM
from NeuralNetworks.RNN import RNN
from utils import *
from normalize import Normalize

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
# prevent caching so website can be updated dynamically
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

def lstm_predict(stock, start, end):
    # shift start date -4 days for correct test/train i/o

    # get stock data
    try:
        df = get_stock_data(stock, start, end, json=False)
    except:
        # error info
        e = sys.exc_info()
        print(e)
        print("lstm predict fail")
        return e

    print("got stock data")

    stock = df
    
    scaler = Normalize(df)
    df = scaler.normalize_data(df)

    print(scaler)

    train_max_index = round((len(df) - 1) * 0.75)
    print(len(df), train_max_index)

    training_input_1 = [[df[i-6], df[i-5]] for i in range(6, train_max_index)]
    training_input_2 = [[df[i-4], df[i-3]] for i in range(6, train_max_index)]
    training_input_3 = [[df[i-2], df[i-1]] for i in range(6, train_max_index)]
    target = [[i] for i in df[6:train_max_index]]

    training_input_1 = np.array(training_input_1, dtype=float)
    training_input_2 = np.array(training_input_2, dtype=float)
    training_input_3 = np.array(training_input_3, dtype=float)
    target = np.array(target, dtype=float)

    assert len(training_input_1) == len(training_input_2) == len(training_input_3) == len(target)

    # create neural network
    NN = LSTM()

    # number of training cycles
    training_cycles = 100

    # train the neural network
    for cycle in range(training_cycles):
        for n in training_input_1:
            output = NN.train(training_input_1, training_input_2, training_input_3, target)


    # de-Normalize
    output = scaler.denormalize_data(output)
    target = scaler.denormalize_data(target)

    # transpose
    output = output.T


    # change data type so it can be plotted
    prices = pd.DataFrame(output)

    #print("\nTraining output:\n", output)

    print("\nTraining MSE Accuracy: {:.4f}%".format(100 - mse(target, output)))
    print("Training RMSE Accuracy: {:.4f}%".format(100 - rmse(target, output)))
    print("Training MAPE Accuracy: {:.4f}%".format(100 - mape(target, output)))

    # [price 2 days ago, price yesterday] for each day in range
    testing_input_1 = [[df[i-6], df[i-5]] for i in range(train_max_index, len(df))]
    testing_input_2 = [[df[i-4], df[i-3]] for i in range(train_max_index, len(df))]
    testing_input_3 = [[df[i-2], df[i-1]] for i in range(train_max_index, len(df))]
    test_target = [[i] for i in df[train_max_index:len(df)]]

    assert len(testing_input_1) == len(testing_input_2) == len(testing_input_3) == len(test_target)

    testing_input_1 = np.array(testing_input_1, dtype=float)
    testing_input_2 = np.array(testing_input_2, dtype=float)
    testing_input_3 = np.array(testing_input_3, dtype=float)
    test_target = np.array(test_target, dtype=float)

    #print("\nTest input", input)
    #print("\nTest target output", test_target)

    # test the network with unseen data
    test = NN.test(testing_input_1, testing_input_2, testing_input_3)

    # de-Normalize data
    test = scaler.denormalize_data(test)
    test_target = scaler.denormalize_data(test_target)

    # transplose test results
    test = test.T

    # accuracy
    accuracy = 100 - mape(test_target, test)
    print(accuracy)

    print("returning")
    print(stock, prices, pd.DataFrame(test), str(round(accuracy, 2)), sep="\n")
    return stock, prices, pd.DataFrame(test), str(round(accuracy, 2))


def rnn_predict(stock, start, end):
    # get stock data
    try:
        df = get_stock_data(stock, start, end, json=False)
    except:
        # error info
        e = sys.exc_info()
        print(e)
        print("rnn predict fail")
        return e

    print("got stock data")

    # normalize
    scaler = Normalize(df, max=True)
    normalized = scaler.normalize_data(df)

    print("normalized")

    # get training and testing inputs and outputs
    train_inputs, train_targets, test_inputs, test_targets = train_test_split(normalized)

    print("i/o")

    train_inputs = np.array(train_inputs)
    train_targets = np.array(train_targets)
    test_inputs = np.array(test_inputs)
    test_targets = np.array(test_targets)

    print("np.array")

    # returns 3d array in format [inputs, timesteps, features]
    train_inputs = to_3d(train_inputs)
    test_inputs = to_3d(test_inputs)

    print("3d")

    #print(train_inputs.shape, train_targets.shape)
    #print(test_inputs.shape, test_targets.shape)

    NN = RNN_V2()
    train_outputs = NN.train(train_inputs, train_targets, epochs=100)
    print("trained")
    test_outputs = NN.test(test_inputs)
    print("tested")

    # de-normalize
    train_outputs = scaler.denormalize_data(train_outputs)
    train_targets = scaler.denormalize_data(train_targets)
    test_outputs = scaler.denormalize_data(test_outputs)
    test_targets = scaler.denormalize_data(test_targets).T

    print(test_outputs, test_targets, sep="\ntargets:\n")

    print("denormalized")
    # accuracy
    accuracy = 100 - mape(test_targets, test_outputs)
    print(accuracy)

    print("returning")

    return df, pd.DataFrame(train_outputs), pd.DataFrame(test_outputs), str(round(accuracy, 2))

def train_test_split(df, split=0.75):
    # if split=0.75, splits data into 75% training, 25% test
    # provides targets for training and accuracy measurments
    # -4 necessary as we take in 4 extra input from start date -4
    # to ensure train_input + test_input = len(df) -4
    # meaning the output data is the same size as the original
    max_index = round((len(df) - 1 - 4) * split)

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
    assert len(train_inputs) + len(test_inputs) == len(df) - 4

    return train_inputs, train_targets, test_inputs, test_targets

def shift_date(date, shift=4):
    # y/m/d
    new_date = dt.date(*date) - dt.timedelta(days=shift)
    new_date = new_date.strftime("%Y/%m/%d")
    new_date = [int(s) for s in new_date.split("/")]
    #print(date, new_date)
    return new_date

def handle_nn(stock, start, end, model):
    # create nn the user selected
    if model == "ff":
        NN = FeedForward()
        return predict(stock, start, end, NN)
    if model == "rnn":
        return rnn_predict(stock, start, end)
    else:
        # TODO update nn created below when finished
        return lstm_predict(stock, start, end)

def predict(stock, start, end, NN):
    # shift start date -4 days for correct test/train i/o
    start = shift_date(start)

    # get stock data
    try:
        df = get_stock_data(stock, start, end, json=False)
    except:
        # error info
        e = sys.exc_info()
        print(e)
        print("predict fail")
        return e

    print(" got data")

    # split data into training and testing
    train_inputs, train_targets, test_inputs, test_targets = train_test_split(df)

    print("got i/0")

    # normalize data
    scaler = Normalize(df)
    print(scaler.factor)
    train_inputs = scaler.normalize_data(train_inputs)
    train_targets = scaler.normalize_data(train_targets)

    print("normalized")

    # number of training cycles
    epochs = 100

    # train the neural network
    for epoch in range(epochs):
        for prices in train_inputs:
            train_outputs = NN.train(train_inputs, train_targets)

    print("trained")

    # de-Normalize data
    train_outputs = scaler.denormalize_data(train_outputs)
    prices = pd.DataFrame(train_outputs.T)

    print("denormalized")

    # Normalize data
    test_inputs = scaler.normalize_data(test_inputs)

    print("normalized")

    # test the network with unseen data
    test_outputs = NN.test(test_inputs)

    print("tested")

    # de-Normalize data
    test_inputs = scaler.denormalize_data(test_inputs)
    test_outputs = scaler.denormalize_data(test_outputs)

    print("denormalized")

    # transplose test results
    test_outputs = test_outputs.T

    # accuracy of test prediction
    # rounds accuracy to 2 decimal places
    accuracy = round(100 - mape(test_targets, test_outputs), 2)

    # return original stock data, training output, testing output, test prediction accuracy
    return df[4:], prices, pd.DataFrame(test_outputs), str(accuracy)


def get_stock_data(ticker, start=[2019, 1, 1], end=[2019, 12, 31], json=True):
    # *list passes the values in list as parameters
    start = dt.datetime(*start)
    end = dt.datetime(*end)

    # download csv from yahoo finance
    try:
        df = web.DataReader(ticker, 'yahoo', start, end)
    except:
        # error info
        e = sys.exc_info()
        print(e)
        print("get data fail")
        return e

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
# app routes are urls which facilitate
# data transmit, mainly:
# Get and Post requests
@app.route('/')
def index():
    # load html from templates directory
    return render_template('index.html')

@app.route('/getpythondata')
def get_python_data():
    return get_stock_data("TSLA")

@app.route('/postjsdata', methods=['POST'])
def post_js_data():
    # POST request
    if request.method == 'POST':
        # convert to JSON
        data = request.get_json(force=True)
        stock = data["stock"].upper()
        start = data["startDate"].split("-")
        end = data["endDate"].split("-")
        model = data["model"]

        # convert strings to integers
        start, end = [int(s) for s in start], [int(s) for s in end]

        try:
            # get original stock data, train and test results
            actual, train_res, test_res, accuracy = handle_nn(stock, start, end, model)
            print(accuracy)
        except:
            # error info
            e = sys.exc_info()
            print(e)
            print("handle_nn fail")
            return "error", 404

        # convert pandas dataframe to list
        actual = [i for i in actual]
        train_res, test_res = [i for i in train_res[0][:]], [i for i in test_res[0][:]]

        # range of x values for plotting
        actualX = [i for i in range(len(actual))] 
        trainX = [i for i in range(len(train_res))] 
        testX = [i for i in range(len(train_res), len(train_res) + len(test_res))]

        # connect training and test lines in plot
        test_res.insert(0, train_res[-1])
        testX.insert(0, trainX[-1])

        return {"stock" : stock,
                "actual" : actual,
                "actualX" : actualX,
                "train" : train_res,
                "trainX" : trainX, 
                "test" : test_res,
                "testX" : testX,
                "accuracy" : accuracy}
        
if __name__ == '__main__':
    app.run(host = "0.0.0.0", port = 3000, debug=True)