import sys
from time import sleep
from random import random
from multiprocessing import Pool
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

from pandas_datareader import data as pdr
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas_ta as ta
yf.pdr_override()
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score
from datetime import datetime, timedelta
import pandas as pd
# import functions from utils.py
from utils import getStockData, neural_network, callback, sma_neural_network

# function to create a 20 day SMA
def sma20(stock):
    LOSS_THRESHOLD = 0.015
    ACCURACY_THRESHOLD = -1

    callbacks = callback(ACCURACY_THRESHOLD, LOSS_THRESHOLD, stock)
    yf.pdr_override()
    print("starting machine learning on " + stock)
    ACCURACY_THRESHOLD = 0.999
    LOSS_THRESHOLD = 0.015
    # Mode is the days to look in the future
    modes = [10,20,30,50,100]
    for mode in modes:
        yf.pdr_override()
        startdate = datetime(1980, 1, 1)
        currentdate = datetime.now()
        enddate = datetime(currentdate.year, currentdate.month, currentdate.day)
        data = pdr.get_data_yahoo("^GSPC", start=startdate, end=enddate)
        #  get the VIX data
        vix = pdr.get_data_yahoo("^VIX", start=startdate, end=enddate)
        # get the dow jones data
        dow = pdr.get_data_yahoo("^DJI", start=startdate, end=enddate)
        # get the nasdaq data
        nasdaq = pdr.get_data_yahoo("^IXIC", start=startdate, end=enddate)
        # get the russell 2000 data
        russell = pdr.get_data_yahoo("^RUT", start=startdate, end=enddate)
        # get the United States interest rate data
        us10yr = pdr.get_data_yahoo("^TNX", start=startdate, end=enddate)
        # get the options volume data
        options = pdr.get_data_yahoo("^OEX", start=startdate, end=enddate)
        # remember to make this dynamic
        stockdata = pdr.get_data_yahoo(stock, start=startdate, end=enddate)
        stockdata = stockdata.dropna()

        # # add stock to the data
        data[stock] = stockdata['Adj Close']
        # add the future column to the data

        #  add the dow jones data to the data
        data['Dow'] = dow['Adj Close']
        # add stock High's to the data
        data['StockHigh'] = stockdata['High']
        # add stock Low's to the data
        data['StockLow'] = stockdata['Low']
        # add stock Volume's to the data
        data['StockVolume'] = stockdata['Volume']
        # add stock Open's to the data
        data['StockOpen'] = stockdata['Open']
        # add the nasdaq data to the data
        data['Nasdaq'] = nasdaq['Adj Close']
        # add the russell 2000 data to the data
        data['Russell'] = russell['Adj Close']
        # add the 10 year treasury data to the data
        data['US10YR'] = us10yr['Adj Close']
        # add the options volume data to the data
        data['Options'] = options['Adj Close']
        # add the DXY data to the data
        data['DXY'] = pdr.get_data_yahoo("DX-Y.NYB", start=startdate, end=enddate)['Adj Close']
        # add the

        data['VIX'] = vix['Adj Close']

        # create a rsi and stock variable on data with length of 3, 5, 10, 14, 20, 30, 50, 100, 200
        data['RSI3'] = ta.rsi(data['Adj Close'], length=3)
        data['RSI5'] = ta.rsi(data['Adj Close'], length=5)
        data['RSI10'] = ta.rsi(data['Adj Close'], length=10)
        data['RSI14'] = ta.rsi(data['Adj Close'], length=14)
        data['RSI20'] = ta.rsi(data['Adj Close'], length=20)
        data['RSI30'] = ta.rsi(data['Adj Close'], length=30)
        data['RSI50'] = ta.rsi(data['Adj Close'], length=50)
        data['RSI100'] = ta.rsi(data['Adj Close'], length=100)
        data['RSI200'] = ta.rsi(data['Adj Close'], length=200)

        # do the same for the stocks data
        data['stockRSI3'] = ta.rsi(data[stock], length=3)
        data['stockRSI5'] = ta.rsi(data[stock], length=5)
        data['stockRSI10'] = ta.rsi(data[stock], length=10)
        data['stockRSI14'] = ta.rsi(data[stock], length=14)
        data['stockRSI20'] = ta.rsi(data[stock], length=20)
        data['stockRSI30'] = ta.rsi(data[stock], length=30)
        data['stockRSI50'] = ta.rsi(data[stock], length=50)
        data['stockRSI100'] = ta.rsi(data[stock], length=100)
        data['stockRSI200'] = ta.rsi(data[stock], length=200)

        # create a stochastic oscillator and stock variable on data with length of 3, 5, 10, 14, 20, 30, 50, 100, 200
        data['Stoch3k'] = ta.stoch(data['High'], data['Low'], data['Adj Close'], length=3)['STOCHk_14_3_3']
        data['Stoch3d'] = ta.stoch(data['High'], data['Low'], data['Adj Close'], length=3)['STOCHd_14_3_3']

        #  add the 3, 5, 10, 14, 20,50,100 SMA to the data
        data['SMA3'] = data['Adj Close'].rolling(window=3).mean()
        data['SMA5'] = data['Adj Close'].rolling(window=5).mean()
        data['SMA10'] = data['Adj Close'].rolling(window=10).mean()
        data['SMA14'] = data['Adj Close'].rolling(window=14).mean()
        data['SMA20'] = data['Adj Close'].rolling(window=20).mean()
        data['SMA50'] = data['Adj Close'].rolling(window=50).mean()
        data['SMA100'] = data['Adj Close'].rolling(window=100).mean()
        data['SMA200'] = data['Adj Close'].rolling(window=200).mean()

        #  <--------- CHOOSE THE DATA THAT WE WANT TO PREDICT --------->

        # <--------- CREATE THE MODEL --------->

        data = data.dropna()

        scaler = MinMaxScaler()
        data2 = scaler.fit_transform(data)
        X = data2

        # load the other models
        model3dayReversal = tf.keras.models.load_model("models/" + stock + '3dayReversal.h5')
        model3dayBull = tf.keras.models.load_model("models/" + stock + '3dayBull.h5')
        model5dayReversal = tf.keras.models.load_model("models/" + stock + '5dayReversal.h5')
        model5dayBull = tf.keras.models.load_model("models/" + stock + '5dayBull.h5')
        model7dayReversal = tf.keras.models.load_model("models/" + stock + '7dayReversal.h5')
        model7dayBull = tf.keras.models.load_model("models/" + stock + '7dayBull.h5')
        model10dayReversal = tf.keras.models.load_model("models/" + stock + '10dayReversal.h5')
        model10dayBull = tf.keras.models.load_model("models/" + stock + '10dayBull.h5')
        model30dayReversal = tf.keras.models.load_model("models/" + stock + '30dayReversal.h5')
        model30dayBull = tf.keras.models.load_model("models/" + stock + '30dayBull.h5')


        data['3dayReversal'] = model3dayReversal.predict(X)
        data['3dayBull'] = model3dayBull.predict(X)
        data['5dayReversal'] = model5dayReversal.predict(X)
        data['5dayBull'] = model5dayBull.predict(X)
        data['7dayReversal'] = model7dayReversal.predict(X)
        data['7dayBull'] = model7dayBull.predict(X)
        data['10dayReversal'] = model10dayReversal.predict(X)
        data['10dayBull'] = model10dayBull.predict(X)
        data['30dayReversal'] = model30dayReversal.predict(X)
        data['30dayBull'] = model30dayBull.predict(X)

        # Load the sma's to make predictions on
        sma = data[stock].rolling(window=mode).mean()
        data['futureSMA'] = sma.shift(-30)
        data = data.dropna()

        # make variable X the data without the last column
        X = scaler.fit_transform( data.iloc[:, :-1])
        # make y the last column
        y = data.iloc[:, -1]

        # Make the neural net
        model = sma_neural_network(X.shape[1])
        # split the data into training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.03, random_state=42)
        # fit the model
        model.fit(X_train, y_train, epochs=3000, batch_size=256, verbose=1, callbacks= [callbacks])
        model.save("smaModels/" + stock + "sma" + str(mode) + ".h5")


if __name__ == '__main__':
    # load the args
    args = sys.argv[1:]
    # first arg is the mode (create or retrain)
    mode = args[0]
    # second arg is the stock name
    stockname = args[1]
    if mode == 'sma':
        sma20(stockname)
    else:
        print("Invalid mode")


