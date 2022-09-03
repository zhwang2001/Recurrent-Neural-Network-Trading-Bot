#Time series forecasting with a Recurrent Neural Network

This trading bot was an attempt at building a passive source of income through options trading on the stock market
Over time i've added higher quality indicator data to my training datasets and the cost functions have decreased
However this bot is missing 1 crucial piece of data that will greatly enhance its performance
Market depth / level 2 data has been well documented in research and at trading firms as having strong predictive power
This is due to the fact that the information contained in the order books are unfilled limit orders. If there is an inbalance
between buyers and sellers the order books will reflect it and can be an early indicator of price movement before it actually occurs.
This is because once those unfilled orders become filled the stock will move in the direction of the imbalance, thus the trading bot
would grealy benefit from this data.

Unfortunately a market depth subscription from IBKR is extremely expensive...
I will definetely try to collect this data in the future once I can afford it. I'm certain that the bot will be 
more precise once I harvest that level 2 data.


Some information on the bot...
 - Trained on tech stocks
 - Utilizes IBKR to make trades
 - Built on Tensorflow and Keras
 - Incorporates LSTM layers to prevent gradient vanishing
 - 6 months of IBKR historical data at a 1 second interval harvested for use in training
 - Mean absolute error = $6.54 on Tesla stock 1 hour into future (On average predictions are $6.50 off from target price)

##NOTE
 - Config.keras files are for the LSTM layers
 - Included the famous F12010 dataset from the nordic stock exchange
 - Other datasets are available to play around with

##Requirements 
 - Must sign up for an IBKR pro account with options trading enabled
 - Only functional with nvidia cuda gpus. Will not work on cpus
 - when running main.py or RNN_regression_time_series.py make sure your current directory is in the folder PlexisBot
 ex. C://Users/charl/Recurrent-Neural-Network-Trading-Bot/PlexisBot

