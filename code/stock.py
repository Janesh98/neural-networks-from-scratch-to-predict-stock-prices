from os import listdir
import pandas as pd
import datetime as dt
from matplotlib import style
import matplotlib.pyplot as plt
import pandas_datareader.data as web

def get_stock_data(ticker):
    csv = ticker + ".csv"

    # load csv file if in current directory
    if csv in listdir(path=".\stock_data_csv\/"):
        df = pd.read_csv(".\stock_data_csv\/" + csv)
        #print(df)
    
    # download csv from yahoo finance
    else:
        start = dt.datetime(2019, 1, 1)
        end = dt.datetime(2019, 12, 31)
        df = web.DataReader(ticker, 'yahoo', start, end)
        #print(df.tail())
        # save csv file
        df.to_csv(".\stock_data_csv\/" + csv)
        
    return df

def plot_stock(df):
    df['Adj Close'].plot()
    plt.show()

def main():
    style.use('ggplot')
    ticker = "TSLA"
    df = get_stock_data(ticker)
    plot_stock(df)

if __name__ == "__main__":
    main()