from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import json
import pandas_datareader.data as web
import datetime as dt
from os import listdir

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


def get_stock_data(ticker="TSLA"):
    csv = ticker + ".csv"

    # load csv file if in current directory
    if csv in listdir(path="..\stock_data_csv\/"):
        df = pd.read_csv("..\stock_data_csv\/" + csv)
        #print(df)
    
    # download csv from yahoo finance
    else:
        start = dt.datetime(2019, 1, 1)
        end = dt.datetime(2019, 12, 31)
        df = web.DataReader(ticker, 'yahoo', start, end)
        #print(df.tail())
        # save csv file
        df.to_csv("..\stock_data_csv\/" + csv)
        
    df = df["Adj Close"]
    return df.to_json()

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
        print(request.get_json(force=True))  # parse as JSON
        return 'OK', 200

    # GET request
    else:
        message = get_stock_data("GOOG")
        return message  # serialize and use JSON headers

def main():
    app.run(host = "0.0.0.0", port = 3000, debug=True)

if __name__ == '__main__':
    main()
