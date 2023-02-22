import numpy as np
import pandas_ta as ta
import yfinance as yf
from pandas_datareader import data as pdr
from scipy.stats import  binom_test
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
yf.pdr_override()


# import early stopping

#function to retrieve the data from the yahoo finance api
# Parameters:
#   ticker: the ticker symbol of the stock
#   mode: an array of form [dayDelta, +- %]

def getStockData(stock, mode):
    startdate = datetime(1900, 1, 1)
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
    stockdata["FutureDown"] = np.where (stockdata["Adj Close"].shift(-mode[0]) < stockdata["Adj Close"] * (1 - mode[1]), 1, 0)
    stockdata["FutureUp"] = np.where (stockdata["Adj Close"].shift(-mode[0]) > stockdata["Adj Close"] * (1 + mode[1]), 1, 0)
    print(stockdata["FutureDown"])

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

    # create deriviatives for all the SMA's set the
    data['SMA3Derivative'] = data['SMA3'].diff(periods=3)
    data['SMA5Derivative'] = data['SMA5'].diff(periods=5)
    data['SMA10Derivative'] = data['SMA10'].diff(periods=10)
    data['SMA14Derivative'] = data['SMA14'].diff(periods=14)
    data['SMA20Derivative'] = data['SMA20'].diff(periods=20)
    data['SMA50Derivative'] = data['SMA50'].diff(periods=50)
    data['SMA100Derivative'] = data['SMA100'].diff(periods=100)
    data['SMA200Derivative'] = data['SMA200'].diff(periods=200)

    # create a derivative for the stockRSI
    data['stockRSI3Derivative'] = data['stockRSI3'].diff(periods=3)
    data['stockRSI5Derivative'] = data['stockRSI5'].diff(periods=5)
    data['stockRSI10Derivative'] = data['stockRSI10'].diff(periods=10)
    data['stockRSI14Derivative'] = data['stockRSI14'].diff(periods=14)
    data['stockRSI20Derivative'] = data['stockRSI20'].diff(periods=20)
    data['stockRSI30Derivative'] = data['stockRSI30'].diff(periods=30)
    data['stockRSI50Derivative'] = data['stockRSI50'].diff(periods=50)
    data['stockRSI100Derivative'] = data['stockRSI100'].diff(periods=100)
    data['stockRSI200Derivative'] = data['stockRSI200'].diff(periods=200)








    #  <--------- CHOOSE THE DATA THAT WE WANT TO PREDICT --------->

    # <--------- CREATE THE MODEL --------->

    # shave the data to only include the rows that have a value for all the variables
    data[stock + 'FutureUp'] = stockdata['FutureUp']
    data[stock + 'FutureDown'] = stockdata['FutureDown']
    data = data.dropna()
    y1 = data[stock + 'FutureUp']
    y2 = data[stock + 'FutureDown']
    data = data.drop([stock + 'FutureUp', stock + 'FutureDown'], axis=1)
    print(data.head())

    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    return data, y1, y2

# Function to create the neural network
# parameters:
#   input_dim: the dimension of inputs
def neural_network(input_dim):
    # make learning rate optimizer
    # create model
    # make model weights to reduce false positives
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(512, activation='relu', input_dim=input_dim))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(8, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def sma_neural_network(input_dim):
    # create model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(512, activation='relu', input_dim=input_dim))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(8, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model

def callback(ACCURACY_THRESHOLD, LOSS_THRESHOLD, stock):
# class to stop training when a monitored quantity has reached accuracy threshold
    class myCallback(tf.keras.callbacks.Callback):
         def on_epoch_end(self, epoch, logs={}):
            if(logs.get('val_accuracy')>= ACCURACY_THRESHOLD and logs.get('val_loss') <= LOSS_THRESHOLD):
                print("FINISHED " + stock)
                self.model.stop_training = True
    # create the callback
    callbacks = myCallback()
    return callbacks

def early_stop():
    # create the early stop callback
    # stop the model from training once the accuracy starts to decrease
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=150)
    return early_stop

def optimize_threshold(predictions, actual_values):
    # find a threshold value so that the p value for the true positives is
    # 0.6
    # initialize the threshold
    best_threshold = 0
    # initialize the best p value
    best_p = 0
    # iterate through the predictions
    for i in range(900):
    #    calculate the confusion matrix using the make confusion matrix function
        confusion_matrix = make_confusion_matrix(predictions, actual_values, i/1000)
    #    calculate the p value for the true positives
        p = binom_test(confusion_matrix[0][0], confusion_matrix[0][0] + confusion_matrix[0][1], 0.5)
    #    if the p value is greater than the best p value
        if p > best_p:
    #       if the p value reaches 0.9 return the threshold
            if p > 0.6:
                return i/1000
    #       set the best p value to the p value
            best_p = p
    #       set the best threshold to the threshold
            best_threshold = i/1000
    # return the best threshold
    return best_threshold


def make_confusion_matrix(predictions, actual_values, threshold):
    # initialize the confusion matrix
    confusion_matrix = np.zeros((2, 2))
    # iterate through the predictions
    for i in range(len(predictions)):
        # if the prediction is greater than the threshold
        if predictions[i] > threshold:
            # if the actual value is 1
            if actual_values[i] == 1:
                # increment the true positive
                confusion_matrix[0][0] += 1
            # if the actual value is 0
            else:
                # increment the false positive
                confusion_matrix[0][1] += 1
        # if the prediction is less than the threshold
        else:
            # if the actual value is 1
            if actual_values[i] == 1:
                # increment the false negative
                confusion_matrix[1][0] += 1
            # if the actual value is 0
            else:
                # increment the true negative
                confusion_matrix[1][1] += 1
    # return the confusion matrix
    return confusion_matrix

# calculate the deriviative of a list of numbers, with n being the length in the past to use
def derivative(data, n):
    # initialize the derivative list
    derivative = []
    # iterate through the data
    for i in range(len(data)):
        # if the data is not at the beginning
        if i >= n:
            # calculate the derivative
            derivative.append(data[i] - data[i - n])
        # if the data is at the beginning
        else:
            # append a 0
            derivative.append(0)
    # return the derivative
    return derivative












