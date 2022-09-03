from ob import *
import os
import pandas as pd
from ib_insync import *
from ibapi import *
from pandas import *
import time as t
import csv
import talib as ta
import numpy as np
import matplotlib.pyplot as plot
from talib import *


class construct(stream, contractOrders):

    def __init__(self):
        # Specifify the port number
        try:
            # port = input("Enter Port Number Here: ")
            self.ib = IB()
            self.ib.connect('127.0.0.1', 7497, clientId=1)
        except:
            print("Error: Failed to Establish Remote Connection")
            print("IB connection estabilished?: ", self.ib.isConnected())
            exit()

        self.account = "DU5541128"
        self.contID = 76792991  # Tesla
        self.arcontract = Stock(
            symbol='TSLA', exchange='SMART', currency='USD')
        self.contract3 = Stock(
            symbol='TSLA', exchange='SMART', currency='USD')
        self.buyorder = MarketOrder("Buy", 1)
        self.sellorder = MarketOrder("Sell", 1)
        # initiate the list with very first date
        self.date = str(self.date_generator()[0])
        self.current_date = [self.date]

    def date_generator(self, start='20220105', end='20220802'):
        self.start = start
        self.end = end
        self.date_list = []
        datetimes = pd.date_range(start=self.start, end=self.end)
        # converts 2022-06-02 00:00:00 --> 20220602
        for date in datetimes:
            self.date_list.append(str(date.date()).replace('-', ''))
        return self.date_list

    def get_start_date(self):
        return self.start

    def get_end_date(self):
        return self.end

    def hours_to_min(self, start_time):
        hours_list = start_time.split(":")
        _hours = int(hours_list[0]) * 60
        _minutes = int(hours_list[1])
        _seconds = (int(hours_list[2]) / 60)
        __conversions = _hours + _minutes + _seconds

        def __time_interval(minutes=__conversions, interval=1800):
            self.interval = interval
            __modification = minutes + interval / 60

            def __min_to_hour(minute=__modification):
                time_mod = "{:.6f}".format(int(minute) / 60)
                time_list = str(time_mod).split('.')
                time_list[1] = float(time_list[1]) / 1000000 * 60
                second_split = str("{:.2f}".format(time_list[1])).split('.')
                second_split[1] = float(second_split[1]) / 100 * 60
                global formatted_time, price_list
                formatted_time = (
                    f'{time_list[0]}:{"{:02d}".format(int(second_split[0]))}:{"{:02d}".format(int(second_split[1]))}')

                # get the most recently used date
                index = self.date_list.index(self.current_date[-1])
                # iterate through list

                if formatted_time == '12:30:00':  # if it is 12:30 then use future dateJ
                    date_index = self.date_generator()[index + 1]
                    date_index = str(date_index)
                    self.current_date.append(date_index)
                else:  # if it isn't 12:30 then use current date
                    # converts 2022-06-02 00:00:00 --> 20220602
                    date_index = self.date_generator()[index]
                    date_index = str(date_index)

                print(
                    f"Historical data indexed at: {date_index}, {formatted_time}")

                bars = self.ib.reqHistoricalData(
                    contract=self.contract3,
                    endDateTime=f"{date_index} {formatted_time}",
                    durationStr=f"{interval} S",
                    barSizeSetting="1 secs",
                    whatToShow='TRADES',
                    useRTH=True,
                    formatDate=1,
                    keepUpToDate=False)
                self.dfg = util.df(bars)
                _time = [_time for _time in self.dfg.iloc[:, 0]]
                _close = [close_price for close_price in self.dfg.iloc[:, 4]]
                _open = [open_price for open_price in self.dfg.iloc[:, 1]]
                _high = [high_price for high_price in self.dfg.iloc[:, 2]]
                _low = [low_price for low_price in self.dfg.iloc[:, 3]]
                _volume = [volume for volume in self.dfg.iloc[:, 5]]
                price_list = [_time, _close, _open, _high, _low, _volume]
                return [price_list, formatted_time, interval]
            return __min_to_hour()
        return __time_interval()

    def server_harvest(self):
        """
        farm the time, open, high, low, close prices and volume
        """
        # add the level 2 market detph
        # nested for loop practice

        # kijiji
        # school fees

        with open("DATAVALUES.CSV", 'a') as writecsv:
            while str(self.date_generator()[0][-1]) != str(self.get_end_date()):
                self.hours_to_min(
                    start_time='9:30:00')  # start time
                while formatted_time != '12:30:00':  # end time
                    # close_data = self.liveStream(self.contract3, 6, durationstring='60 S',
                    # barSizeSettings='1 secs', datafarm=True)
                    # print(close_data.astype("float64"))
                    final_list = []
                    for count in range(len(price_list[0])):
                        data_list = []
                        for i in range(6):
                            data_list.append(price_list[i][count])
                        final_list.append(data_list)
                    csv_writer = csv.writer(
                        writecsv, delimiter=',', lineterminator="\n")  # line terminator tells writer to endline at newline
                    csv_writer.writerows(final_list)
                    t.sleep(25)
                    self.hours_to_min(start_time=formatted_time)
            print("CSV file has finished updating")

    def writer(self, column=int, write=bool):
        """
        Args:
            column:  return the specific columns from the csv files
            write: choose to append to csv file
        """
        self.write = write
        file_name = os.path.join("DATAVALUES.csv")

        with open(file_name) as file:
            data = file.read()

        print("Volatility Data: ", self.volatility_data())
        rows = data.split("\n")
        self.lines = rows[1:]

        extracted_data = []
        for units in (self.lines):
            units = units.split(',')
            extracted_data.append(units[column])
        # convert into float and numpy array to be processsed by talib
        float_data = [float(x) for x in extracted_data]
        __formated_data = np.array(float_data)

        if self.write == True:
            with open("v4.csv", "a") as self.csv_file:

                rsi = self.indicators('rsi', visualizer=False)
                obv = self.indicators('obv', visualizer=False)
                # adx did not improve performance
                adx = self.indicators('adx', visualizer=False)
                di = self.indicators('di', visualizer=False)
                sma = self.indicators('sma', visualizer=False)

                lines_list = list()
                for number, units in enumerate(self.lines):
                    tup = units, sma[number], rsi[number], obv[number], di[number]
                    lines_list.append((tup))

                print(lines_list)
                print(len(lines_list))
                writer = csv.writer(
                    self.csv_file, delimiter=',', lineterminator="\n")
                writer.writerows(lines_list)
        else:
            return __formated_data

    def indicators(self, indicator=str, visualizer=bool):
        """
        Graph visualizer for various indicators
        also returns specific indiactor values
        Args
            "obv"
            "sma"
            "rsi"
            "all"
        """
        # add matplotlib for validation performance over time
        close_price = self.writer(4, False)
        volume_data = self.writer(5, False)
        high_price = self.writer(2, False)
        low_price = self.writer(3, False)

        # "nan': "lookback" period (a required number of observations before an output is generated)
        obv = ta.OBV(close_price, volume_data)
        sma = ta.SMA(close_price, 2)
        rsi = ta.RSI(close_price, timeperiod=14)
        stof, stos = ta.STOCH(high_price, low_price, close_price, fastk_period=5,
                              slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        adx = ta.ADX(high_price, low_price, close_price, timeperiod=14)
        upperband, middleband, lowerband = ta.BBANDS(
            close_price, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
        pos = ta.PLUS_DI(high_price, low_price, close_price, timeperiod=14)
        neg = ta.MINUS_DI(high_price, low_price, close_price, timeperiod=14)
        di = (pos - neg)

        options = {"obv": obv, "sma": sma, "rsi": rsi, "adx": adx, 'di': di}
        if visualizer == False:
            __choice = options[indicator]
            return __choice

        else:
            __choice = options[indicator]
            print(__choice)
            plot.title(f'{indicator} indicator')
            plot.xlabel("time")
            plot.ylabel(f'{indicator}')
            plot.plot(range(len(close_price)), __choice)
            plot.show()


if __name__ == "__main__":

    taf = t.localtime()
    current_time = t.strftime("%H:%M:%S", taf)
    print("Current_time: ", current_time)
    c = construct()
    c.indicators("di", True)
    c.writer(1, True)

    quit()
    #start_date = '20220101'
    #end_date = '20220802'
    #print("harvesting from", start_date, "to", end_date)
    #c.date_generator(start=start_date, end=end_date)
    #c.server_harvest()

