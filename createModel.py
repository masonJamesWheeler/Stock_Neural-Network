import sys

import numpy as np
import pandas_ta as ta
import yfinance as yf
from matplotlib import pyplot as plt
from pandas_datareader import data as pdr

yf.pdr_override()
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from datetime import datetime
import pandas as pd
# import functions from utils.py
from utils import getStockData, neural_network, callback, early_stop
from createImages import createImage

def createModels (stockName):
    # added code 02/16/23
    # arrays to hold the best values for the confusion matrix's
    modes = []
    three_day = [3, 0.035]
    five_day = [5, 0.065]
    seven_day = [7, 0.08]
    ten_day = [10, 0.12]
    twentyone_day = [21, 0.15]
    modes.append(three_day)
    modes.append(five_day)
    modes.append(seven_day)
    modes.append(ten_day)
    modes.append(twentyone_day)

    print("starting machine learning on " + stockName)
    ACCURACY_THRESHOLD = 1.00
    LOSS_THRESHOLD = 0.01

    #  for each stock in stocks
    stocks = [stockName]
    for stock in stocks:
        for mode in modes:
            for i in range(0, 10):
                bestModel = tf.keras.callbacks.ModelCheckpoint('models/' + str(stock) + str(mode[0]) + 'dayBull.h5', monitor='val_accuracy', mode='max',save_best_only=True)
                # get the data for the stock
                X, y1, y2 = getStockData(stock, mode)

                # print the name of columns of X and y
                # generate a random number between 1-50
                rand = np.random.randint(1, 50)

                X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.8, random_state=rand)
                callbacks = callback(ACCURACY_THRESHOLD, LOSS_THRESHOLD, stock)
                # if the model exists at the path then load the model and add the callback
                try:
                    model = tf.keras.models.load_model('models/' + str(stock) + str(mode[0]) + 'dayBull.h5')
                    print("model loaded " + str(stock) + str(mode[0]) + 'dayBull.h5')
                except:
                    model = neural_network(X.shape[1])
                model.fit(X_train, y_train, epochs=5, batch_size=256, validation_data=(X_test, y_test), callbacks=[callbacks, bestModel])
                # make the model the best model

                bestModel = tf.keras.callbacks.ModelCheckpoint('models/' + str(stock) + str(mode[0]) + 'dayReversal.h5' ,monitor='val_accuracy', mode='max', save_best_only=True)

                X_train, X_test, y2_train, y2_test = train_test_split(X, y2, test_size=0.8, random_state=rand)
                try:
                    model2 = tf.keras.models.load_model('models/' + str(stock) + str(mode[0]) + 'dayReversal.h5')
                    print("model loaded" + str(stock) + str(mode[0]) + 'dayReversal.h5')
                except:
                    model2 = neural_network(X.shape[1])
                model2.fit(X_train, y2_train, batch_size=256, validation_data=(X_test, y2_test), epochs=5, callbacks=[callbacks, bestModel])

                data = []
            print("FINISHED " + stock + " " + str(mode[0]) + " day model")

#       make the images
        createImage(stock)


def predictPrice(stock):
    # SET ACCURACY THERESHOLD
    ACCURACY_THRESHOLD = 0.999
    LOSS_THRESHOLD = 0.001

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
    stockdata["FuturePrice"] = stockdata["Adj Close"].shift(-3)
    stockdata = stockdata.dropna()


    # add stock to the data
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

    # make x all the columns except first column
    data['FuturePrice'] = stockdata['FuturePrice']
    X = pd.DataFrame(data2[:, 1:-1])
    # make the y the last column
    y = pd.DataFrame(data2[:, -1])

    # code to load the models and then run it on the current data
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
    data['21dayReversal'] = model30dayReversal.predict(X)
    data['21dayBull'] = model30dayBull.predict(X)
    data = data.dropna()
    data3 = MinMaxScaler().fit_transform(data)
    data['FuturePrice'] = stockdata['FuturePrice']

    X = pd.DataFrame(data3[:, 1:-1])
    y = pd.DataFrame(data3[:, -1])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.03)

    # class to stop training when a monitored quantity has reached accuracy threshold
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('accuracy') >= ACCURACY_THRESHOLD and logs.get('loss') <= LOSS_THRESHOLD):
                print("FINISHED " + stock)
                self.model.stop_training = True

    # create the callback
    callbacks = myCallback()

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(512, activation='relu', input_dim=45))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(8, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=3000, batch_size=256, validation_data=(X_train, y_train), callbacks=[callbacks])

# create megamodel
def createMegamodel():
    stocks = ["AAPL", "AMZN", "GOOGL", "MSFT", "NFLX", "TSLA", "NVDA", "PYPL", "ADBE", "INTC", "CSCO", "CMCSA", "PEP", "AVGO", "QCOM", "TXN", "COST", "AMGN", "AMAT", "INTU", "GILD", "ADP", "MU", "BKNG", "MDLZ", "FISV", "SBUX", "BIIB", "CHTR", "ISRG", "ILMN", "JD", "MELI", "MNST", "NEE", "NXPI", "ORLY", "PCAR", "REGN", "ROST", "SIRI",
              "SNPS", "TMUS", "VRTX", "WBA", "WDAY", "XEL", "ZM"]
    totaldata = pd.DataFrame()
    modes = []
    three_day = [3, 0.035]
    five_day = [5, 0.065]
    seven_day = [7, 0.08]
    ten_day = [10, 0.12]
    twentyone_day = [21, 0.15]
    modes.append(three_day)
    modes.append(five_day)
    modes.append(seven_day)
    modes.append(ten_day)
    modes.append(twentyone_day)
#     iterate through the stocks and conctenate the data into one dataframe
    for mode in modes:
        for stock in stocks:
            currdata = getStockData(stock, mode)
            # turn currdata into a dataframe
            currdata = pd.DataFrame(currdata)
            totaldata = pd.concat([totaldata, currdata])
        totaldata = totaldata.dropna()
        scaler = MinMaxScaler()
        data2 = scaler.fit_transform(totaldata)
        # make x all the columns except first column and the last column
        X = pd.DataFrame(data2[:, 1:-1])
        # make the y the last column
        y = pd.DataFrame(data2[:, -1])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.03, random_state=42)
        model = neural_network(X.shape[1])
        model.fit(X_train, y_train, epochs=100, batch_size=128, validation_data=(X_test, y_test))
        model.save("models/" + "mega" + str(mode[0]) + "day.h5")


    # make x all the columns except first column and the last column




if __name__ == '__main__':
    # load the args
    args = sys.argv[1:]
    # first arg is the mode (create or retrain)
    mode = args[0]
    # if args < 2 then run megamodel
    if len(args) < 2:
        createMegamodel()
    else:
        # second arg is the stock name
        stockname = args[1]
        if mode == 'learn':
            createModels(stockname)
        else:
            print("Invalid mode")
