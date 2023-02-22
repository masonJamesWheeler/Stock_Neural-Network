Stock Prediction Neural Network

This program implements a neural network to make predictions on stock prices. The neural network is trained on historical stock data and uses a variety of technical indicators to predict future stock prices. The program consists of several files that perform different functions:

Concept:

The larger concept of this program is that while past stock patterns are not a guarantee of future stock movements,
there are psychological factors which lead to certain volatile movements. This program seeks to recognize situations of 
euphoria and fear in the market, through various technical indicators mainly: derivatives of more common indicators. And predict the future stock price based on these.
Further, the program attacks a problem that is largely overlooked traditionally in stock models: overfitting on past data.
We solve this by making fast random shuffles to the data forcing the model to be nimble and non-biased. 


Files:

createModel.py: This file defines the architecture of the neural network and trains it on the preprocessed data that is fetched through the utility functions in utils.py.

createImages.py: After creation of the neural network, this file is used to create the output images for the user to see the predictions made by the neural network.

smaModel.py: This file defines a different kind of prediction model where the neural network is used to create a forward projection of the Simple Moving Average (SMA) of the stock price. This model is used to compare the performance of the neural network to a simpler model.

utils.py: This file contains utility functions used by the other files. notably the functions of retrieving stock data from Yahoo Finance and preprocessing the data. As well as setting thresholds for the models predictions based on adjustable p-values.

stocklist.txt: This file contains a list of stock symbols that are adjustable by the user. The neural network is trained on the stock data of these symbols and serves as an input for the automated shell scripts.

updateModels: This shell script updates the stock data to the most recent data and re-trains the neural network on the updated data.

runimages: This shell script runs the pre-processing of the historical stock data and creates the output images for the user to see the predictions made by the neural network.

Usage
To use the program, follow these steps:

Clone the repository to your local machine.

Install the required dependencies by running pip install -r requirements.txt.

Update the stocklist.txt file with the list of stock symbols you want to use for training and testing the neural network.

Run the updateModels shell script to update the stock data and re-train the neural network on the updated data.

Run the runimages shell script to preprocess the historical stock data and create the output images based on the most recent model and the most recent data.

Run the createModel.py file to define the neural network architecture and train it on pre-proccessed data.
I.e. python createModel.py learn AAPL

(Optional) Run the smaModel.py file to define a simpler model based on SMAs and compare its performance to the neural network.
I.e. python smaModel.py sma AAPL

Disclaimer
This program is for educational purposes only and should not be used for actual trading. The predictions made by the neural network may not be accurate and should not be relied upon for making financial decisions.