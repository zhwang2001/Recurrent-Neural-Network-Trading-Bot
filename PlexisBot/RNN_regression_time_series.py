import os
import this
import numpy as np
import pandas as pd
from time import time
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plot
from keras.datasets import boston_housing

from tensorflow.python.keras import Sequential, layers, Input, Model, callbacks, optimizers
from tensorflow.python.keras.optimizer_v2.adam import Adam


# TODO:
# Utilize level 2 data
# Smoothing
# Min max normalization
# additional indicators for v3
# Nested dictionary for configs
# prediction[1]
# find optimal sequence length and delay
# should i preprocess obv and level 2(inbalance)
# dataset from lse
# try beta and other inidactors in rob the quants video
# try voliatility indicators work
# try different sampling rates
# Regularization
# preprocess obv slope?
# index layers for feature extraction


# FIXME:
# server_harvest() can't harvest data during weekends
# rewriting csv file (write()) adds apostrophe (could add another csv_write to scan for apostrophe and remove)


class recurrent_neural_net:
    """
    Used to process time series, use in combination with scalar regression

    Primarily uses these indicators to perform price forecasting
    - rsi (momentum / trending indicator) - optimized for trading ranges
    - obv (volume / momentum indicator) - optimized for trending markets
    - beta (statistics indicator)
    - level 2 data (inbalanace detection)
    - CDLDOJISTAR  (pattern recognition)

    Saved Models:

    config1.keras:
    1. optimizer = RMSprop
    2. metrics = mae
    3. loss function = mse

    config2.keras:
    1. optimizer = Adam
    2. metrics = mae
    3. loss function = categorical crossentropy

    """

    def __init__(self):
        """Preparation of file"""
        file_name = os.path.join("v4.csv")

        with open(file_name) as file:
            data = file.read()

        rows = data.split("\n")  # (everything append each row to a list
        # index the first row (header) and use comma seperator
        self.header = rows[0].split(",")
        # every other row after (header) will be features
        self.features = rows[1:]
        print(self.header)
        print("header length: ", len(self.header))  # 15 lines
        print("number of rows: ", len(self.features))  # 420451 lines

    def __data_prep(self):
        """prepare data to feed into recurrent neural net"""

        # prepare some data for visualizer()
        global temperature, raw_data, sequence_length

        # Parsing data
        # the goal is to trasnfer column 2 (temperature) in the temperature array
        # everything else including temperature but not including date time column is stored in raw data
        temperature = np.zeros((len(self.features),))  # now shape is (420451,)
        print(temperature.shape)
        raw_data = np.zeros((len(self.features), len(self.header) - 1))
        print(raw_data.shape)

        for i, line in enumerate(self.features):
            values = [float(column) for column in line.split(',')[1:]]
            temperature[i] = values[4]
            raw_data[i, :] = values[:]

        # Prepare some data for training()
        # 25% for testing, 25% validation, 50% training
        num_train_samples = int(0.50 * len(raw_data))
        num_val_samples = int(0.25 * len(raw_data))
        num_test_samples = int(
            len(raw_data) - num_val_samples - num_train_samples)

        print(
            "num_train_samples: ", num_train_samples,
            "\nnum_val_samples: ", num_val_samples,
            "\nnum_test_samples: ", num_test_samples
        )

        # normalize the data
        global mean, std
        mean = raw_data[:num_train_samples].mean(axis=0)
        raw_data -= mean
        std = raw_data[:num_train_samples].std(axis=0)
        raw_data /= std

        # create input and predictive timeseries output
        sampling_rate = 10  # sample 1 point of data every 10 seconds
        sequence_length = 360 # use 360 samples (1 hour total)
        delay = sampling_rate * (sequence_length + 360) #predict 1 hour into future
        #NOTE optimial is 32 timesteps (mins) at a time //INCREASED TO 64 to increase gpu usage
        batch_size = 32

        global train_dataset, val_dataset, test_dataset
        train_dataset = keras.utils.timeseries_dataset_from_array(
            raw_data[:-delay],
            targets=temperature[delay:],
            sampling_rate=sampling_rate,
            sequence_length=sequence_length,
            shuffle=True,
            batch_size=batch_size,
            start_index=0,
            end_index=num_train_samples)

        val_dataset = keras.utils.timeseries_dataset_from_array(
            data=raw_data[:-delay],
            targets=temperature[delay:],
            sampling_rate=sampling_rate,
            sequence_length=sequence_length,
            shuffle=True,
            batch_size=batch_size,
            start_index=num_train_samples,
            end_index=num_train_samples + num_val_samples)

        test_dataset = keras.utils.timeseries_dataset_from_array(
            data=raw_data[:-delay],
            targets=temperature[delay:],
            sampling_rate=sampling_rate,
            sequence_length=sequence_length,
            shuffle=True,
            batch_size=batch_size,
            start_index=num_train_samples + num_val_samples)

        # stats
        for samples, targets in train_dataset:
            print(samples.shape, samples.ndim, samples.dtype)
            print(targets.shape, targets.ndim, targets.dtype)
            print(len(train_dataset))
            print(len(val_dataset))
            print(len(test_dataset))
            break

        inputs = Input(shape=(sequence_length, raw_data.shape[-1]))
        #x = layers.LSTM(18, recurrent_dropout=0.25)(inputs)
        x = layers.CuDNNLSTM(16)(inputs)
        #x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(1)(x)
        model = Model(inputs, outputs)
        callback_list = [
            callbacks.ModelCheckpoint("config4.keras", save_best_only=True)]

        #optimizer = Adam(learning_rate=0.01, clipnorm=1)
        optimizer = Adam()  # aggressive
        model.compile(optimizer=optimizer, loss="mae",
                      metrics=[tf.keras.metrics.RootMeanSquaredError()])
        train_run = model.fit(train_dataset, epochs=10,
                              validation_data=val_dataset, callbacks=callback_list)

        model = keras.models.load_model("config4.keras")
        print(f"Test MAE: {model.evaluate(test_dataset)[1]:.2f}")

        model.summary()
        print(model.weights)
        predictions = model.predict(test_dataset)
        print(predictions[0:10])

        for count, row in enumerate(range(num_val_samples + num_train_samples, num_val_samples + num_train_samples + 10)):
            print('Predicted Value: ', predictions[count][0] * std[3] + mean[3],
                  'Actual Value: ', self.features[row].split(',')[4])
        return train_run

    def performance(self):
        train_run = self.__data_prep()
        print(train_run.history.keys())
        loss = train_run.history["loss"]
        rmse = train_run.history["root_mean_squared_error"]
        val_loss = train_run.history["val_loss"]
        val_rmse = train_run.history["val_root_mean_squared_error"]
        plot.title("Performance")
        plot.ylabel("metrics")
        plot.xlabel("epochs")
        plot.plot(range(1, len(val_loss) + 1), "bo", loss, label="loss")
        plot.plot(range(1, len(val_loss) + 1), "bo", rmse, label="rmse")
        plot.plot(range(1, len(val_loss) + 1), "b", val_loss, label="val_loss")
        plot.plot(range(1, len(val_loss) + 1), "b", val_rmse, label="val_rmse")
        plot.legend()
        plot.show()

    @staticmethod
    def visualizer():
        # temperatures across all recorded time
        plot.plot(range(len(temperature)), temperature,
                  label="temperature recording (7 years)")
        plot.legend()
        plot.xlabel("Recordings")
        plot.ylabel("Temperature")
        plot.show()

        # temperatures across an index of time
        plot.plot(range(1440), temperature[:1440],
                  'bo', label="indexed temperature recording")
        plot.legend()
        plot.xlabel("Recordings")
        plot.ylabel("Temperature")
        plot.show()

    @staticmethod
    def __test__():
        # using timeseries_dataset_from_array() we can guess the next number in the sequence
        # [0, 1, 2] 3
        # [1, 2, 3] 4
        # [2, 3, 4] 5
        # [3, 4, 5] 6
        # [4, 5, 6] 7
        # [5, 6, 7] 8

        int_sequence = np.arange(10)
        dummy_dataset = keras.utils.timeseries_dataset_from_array(
            data=int_sequence[:-3],
            targets=int_sequence[2:],
            sequence_length=2,
            batch_size=2)

        for inputs, targets in dummy_dataset:
            for i in range(inputs.shape[0]):
                print([int(x) for x in inputs[i]], int(targets[i]))


if __name__ == "__main__":
    rnn = recurrent_neural_net()
    rnn.performance()
    quit()
    # rnn = recurrent_neural_net()
    # rnn.__init__()
    # rnn.__data_prep()
    # rnn.__test__()
    # rnn.evaluate_naive_baseline()
    # rnn.visualize_naive_dense()
