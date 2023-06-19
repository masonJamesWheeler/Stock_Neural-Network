# Stock Prediction Neural Network

## Table of Contents

1. [Introduction](#Introduction)
2. [Concept](#Concept)
3. [Directory Structure](#Directory-Structure)
4. [Usage](#Usage)
5. [Disclaimer](#Disclaimer)
6. [Contact](#Contact)

## Introduction

Welcome to the Stock Prediction Neural Network project. This program utilizes a neural network trained on historical stock data, incorporating various technical indicators to predict future stock prices. By identifying market euphoria and fear instances, it offers an innovative approach to predict stock market movements.

## Concept

While past stock patterns can't guarantee future stock movements, psychological factors can trigger certain volatile market movements. This program identifies these situations of market euphoria and fear using various technical indicators, derivatives of more common indicators mainly, and predicts future stock prices based on these. This project also addresses an often overlooked problem in traditional stock models - overfitting on past data, by incorporating random data shuffles to ensure the model's adaptability and non-biased nature.

![AMZN 2 21 2023](https://user-images.githubusercontent.com/106849824/220750417-7ba02a03-3153-4fd8-a73b-e84e865ae666.png)

## Directory Structure

The project directory structure:

```
.
├── createModel.py
├── createImages.py
├── smaModel.py
├── utils.py
├── stocklist.txt
├── updateModels
└── runimages
```

- `createModel.py`: Defines the neural network architecture and trains it on preprocessed data.
- `createImages.py`: Generates output images based on the neural network's predictions.
- `smaModel.py`: Defines an SMA-based prediction model for performance comparison.
- `utils.py`: Contains utility functions used by other files, such as stock data retrieval and preprocessing.
- `stocklist.txt`: A user-adjustable list of stock symbols serving as the neural network's training input.
- `updateModels`: A shell script for updating stock data and re-training the neural network.
- `runimages`: A shell script for pre-processing historical stock data and creating output images.

## Usage

To use the program:

1. Clone the repository.
```bash
git clone <repository_link>
```
2. Install the dependencies.
```bash
pip install -r requirements.txt
```
3. Update `stocklist.txt` with your preferred stock symbols.
4. Run the `updateModels` shell script.
```bash
./updateModels
```
5. Run the `runimages` shell script.
```bash
./runimages
```
6. Run `createModel.py`.
```bash
python createModel.py learn <symbol>
```
7. (Optional) Run `smaModel.py` for SMA-based model performance comparison.
```bash
python smaModel.py sma <symbol>
```

## Disclaimer

This program is for educational purposes only. It shouldn't be used for actual trading. The predictions may not be accurate and shouldn't be relied upon for financial decisions.
