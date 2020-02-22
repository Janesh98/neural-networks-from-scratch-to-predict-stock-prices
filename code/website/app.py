from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import json
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

def predict():
    # returns pandas dataframe
    df = get_stock_data("TSLA", json=False)
    print(df)

    # X = (adjclose for 2 days ago, adjclose for previous day)
    # y = actual adjclose for current day
    X = [[df[i-2], df[i-1]] for i in range(len(df[:102])) if i >= 2]
    y = [[i] for i in df[2:102]]

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)


    assert len(X) == len(y)
    print(len(X), len(y))

    # Normalize
    X = X/1000
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
    y *= 1000

    # transpose
    output = output.T

    # change data type so it can be plotted
    prices = pd.DataFrame(output)

    #print("\nTraining output:\n", output)

    # [price 2 days ago, price yesterday] for each day in range
    input = [[df[i-2], df[i-1]] for i in range(102, 152)]
    test_target = [[i] for i in df[102:152]]

    assert len(input) == len(test_target)

    input = np.array(input, dtype=float)
    test_target = np.array(test_target, dtype=float)

    # Normalize
    input = input/1000

    # test the network with unseen data
    test = NN.test(input)

    # de-Normalize data
    input *= 1000
    test *= 1000

    # transplose test results
    test = test.T

    return prices, pd.DataFrame(test)


def get_stock_data(ticker="TSLA", json=True):
    csv = ticker + ".csv"

    # load csv file if in current directory
    if csv in listdir(path="..\stock_data_csv\/"):
        df = pd.read_csv("..\stock_data_csv\/" + csv)
    
    # download csv from yahoo finance
    else:
        start = dt.datetime(2019, 1, 1)
        end = dt.datetime(2019, 12, 31)
        df = web.DataReader(ticker, 'yahoo', start, end)
        # save csv file
        df.to_csv("..\stock_data_csv\/" + csv)
    
    df = df["Adj Close"]

    # return data as JSON
    if json:
        return df.to_json()

    # return data as csv
    else:
        return df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/getpythondata')
def get_python_data():
    return get_stock_data()

@app.route('/hello', methods=['GET', 'POST'])
def hello():

    # POST request
    if request.method == 'POST':
        print('Incoming..')
        # convert to JSON
        print(request.get_json(force=True))
        return 'OK', 200

    # GET request
    else:
        train_res, test_res = predict()
        # convert pandas dataframe to list
        train_res, test_res = [i for i in train_res[0][:]], [i for i in test_res[0][:]]
        # range of y values for plotting
        trainY = [i for i in range(len(train_res))] 
        testY = [i for i in range(len(train_res) - 1, len(train_res) + len(test_res))]
        return {"train" : train_res,
                "trainY" : trainY, 
                "test" : test_res,
                "testY" : testY}
        
if __name__ == '__main__':
    app.run(host = "0.0.0.0", port = 3000, debug=True)