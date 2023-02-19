import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from pandas_datareader import data as pdr
yf.pdr_override()
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import sys
from utils import getStockData, neural_network, callback, sma_neural_network



def createImage(stockName):
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

    # THIS WILL HOLD STOCK DATA FOR TESLA
    stocks = [stockName]
    #  for each stock in stocks
    for stock in stocks:
        # remember to make this dynamic
        stockdata = pdr.get_data_yahoo(stock, start=startdate, end=enddate)
        stockdata["FutureDown"] = np.where(stockdata['Adj Close'].shift(-5) < stockdata['Adj Close'] * 0.95, 1, 0)
        stockdata["FutureUp"] = np.where(stockdata['Adj Close'].shift(-5) > stockdata['Adj Close'] * 1.05, 1, 0)

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

        # make x all the columns except first column
        X = pd.DataFrame(data2)

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
        data['30dayReversal'] = model30dayReversal.predict(X)
        data['30dayBull'] = model30dayBull.predict(X)

        data["non-normalized_Adj_Close"] = stockdata['Adj Close']
        # reset matplotlib to default
        plt.style.use('ggplot')
        current = plt.figure(figsize=(20, 10), dpi=500)
        # plot the actual stock price
        plt.plot(data[-365:]['non-normalized_Adj_Close'], label='Price', color='#1e40af')
        # set the size of the bubbles by changing the params in the scatter function
        plt.rcParams['lines.markersize'] = 14
        # plot a dot on the Stock price when the reversal probability is greater than 0.95 for the 30 day
        plt.scatter(data[-365:][data['30dayReversal'] > 0.99].index,
                    data[-365:][data['30dayReversal'] > 0.99]['non-normalized_Adj_Close'],
                    label='30 day reversal', color='#881337')
        # plot a dot on the Stock price when the bull probability is greater than 0.9 for the 30 day
        plt.scatter(data[-365:][data['30dayBull'] > 0.9].index,
                    data[-365:][data['30dayBull'] > 0.9]['non-normalized_Adj_Close'],
                    label='30 day bull', color='#4c1d95')

        plt.rcParams['lines.markersize'] = 12
        # plot a dot on the Stock price when the reversal probability is greater than 0.95 for the 10 day
        plt.scatter(data[-365:][data['10dayReversal'] > 0.999].index,
                    data[-365:][data['10dayReversal'] > 0.999]['non-normalized_Adj_Close'],
                    label='10 day reversal', color='#9d174d')
        # plot a dot on the Stock price when the bull probability is greater than 0.9 for the 10 day
        plt.scatter(data[-365:][data['10dayBull'] > 0.9].index,
                    data[-365:][data['10dayBull'] > 0.9]['non-normalized_Adj_Close'],
                    label='10 day bull', color='#2dd4bf')

        plt.rcParams['lines.markersize'] = 10
        # plot a dot on the Stock price when the reversal probability is greater than 0.95 for the 7 day
        plt.scatter(data[-365:][data['7dayReversal'] > 0.999].index,
                    data[-365:][data['7dayReversal'] > 0.999]['non-normalized_Adj_Close'],
                    label='7 day reversal', color='#be185d')
        # plot a dot on the Stock price when the bull probability is greater than 0.9 for the 7 day
        plt.scatter(data[-365:][data['7dayBull'] > 0.9].index,
                    data[-365:][data['7dayBull'] > 0.9]['non-normalized_Adj_Close'],
                    label='7 day bull', color='#0284c7')

        plt.rcParams['lines.markersize'] = 8
        # plot a dot on the Stock price when the reversal probability is greater than 0.95 for the 5 day
        plt.scatter(data[-365:][data['5dayReversal'] > 0.999].index,
                    data[-365:][data['5dayReversal'] > 0.999]['non-normalized_Adj_Close'], label='5 day reversal',
                    color='#b91c1c')
        # plot a dot on the Stock price when the bull probability is greater than 0.9 for the 5 day
        plt.scatter(data[-365:][data['5dayBull'] > 0.9].index,
                    data[-365:][data['5dayBull'] > 0.9]['non-normalized_Adj_Close'],
                    label='5 day bull', color='#047857')

        plt.rcParams['lines.markersize'] = 6
        # plot a dot on the Stock price when the reversal probability is greater than 0.95 for the 3 day
        plt.scatter(data[-365:][data['3dayReversal'] > 0.999].index,
                    data[-365:][data['3dayReversal'] > 0.999]['non-normalized_Adj_Close'], label='3 day reversal',
                    color='#f97316')
        # plot a dot on the Stock price when the bull probability is greater than 0.9 for the 3 day
        plt.scatter(data[-365:][data['3dayBull'] > 0.9].index, data[-365:][data['3dayBull'] > 0.9]['non-normalized_Adj_Close'],
                    label='3 day bull', color='#ec4899')

        # plot the current price on the last data points date as well as the price
        plt.title(stock + ' Price and Predictions')
        plt.xlabel('Date')

        #  below the graph post the last datapoints date and price and round it to 2 decimals
        plt.text(data.index[-1], data['non-normalized_Adj_Close'][-1],
                 str(round(data['non-normalized_Adj_Close'][-1], 2)), fontsize=12)

        plt.ylabel('Price')
        plt.legend(loc='upper left')
        # save the figure to a folder called 'images'
        # make a string of the current date month/day/year
        date = str(datetime.now().month)  + "." + str(datetime.now().day) + "." + str( datetime.now().year)
        plt.savefig('images/' + stock + ' ' + date + '.png')
        plt.close(current)

        # clear the session so that we dont run out of memory
        tf.keras.backend.clear_session()
        # print the stock name so that we know which stock is done
        print(stock)
        # print the last days bulls and reversals
        print(data[-1:][['3dayBull', '3dayReversal', '5dayBull', '5dayReversal', '7dayBull', '7dayReversal', '10dayBull', '10dayReversal', '30dayBull', '30dayReversal']])

def createSma(stock):
    LOSS_THRESHOLD = 0.001
    ACCURACY_THRESHOLD = 0.999

    callbacks = callback(ACCURACY_THRESHOLD, LOSS_THRESHOLD, stock)
    yf.pdr_override()
    print("starting machine learning on " + stock)
    ACCURACY_THRESHOLD = 0.999
    LOSS_THRESHOLD = 0.002
    # Mode is the days to look in the future

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
    full_data = data
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

    data = data.dropna()
    # make variable X the data without the last column
    X = scaler.fit_transform(data)
    modes = [10, 20, 30, 50,100]
    smas=[]
    for mode in modes:
        current_model = tf.keras.models.load_model("smaModels/" + stock + "sma" + str(mode) + '.h5')
#       add the prediction to the data, but shift it into the future by the value of the mode
        currPrediction = current_model.predict(X)
        # print the last 10 values
        print(currPrediction[-10:])
        currPrediction = pd.DataFrame(currPrediction, columns=['sma' + str(mode)])
        currPrediction.index = data.index + pd.DateOffset(days=30)
        smas.append(currPrediction)

#     plot the results
    plt.figure(figsize=(20, 10), dpi=800)
    plt.plot(data[stock][-1000:], label=stock)
    # show the last days stock price in the graph as well as the date
    plt.annotate(str(data[stock][-1]), (data.index[-1], data[stock][-1]))
    for sma in smas:
        plt.plot(sma[-1000:], label='sma' + str(mode))
#     for each sma in the smas array
    plt.legend(loc='upper left')
#     save the image
    plt.savefig('sma_images/' + stock + 'sma.png')









if __name__ == '__main__':
    args = sys.argv[1:]
    mode = args[0]
    if len(args) > 1:
        stock = args[1]
        if mode == 'sma':
            createSma(stock);
    else: createImage(mode)


