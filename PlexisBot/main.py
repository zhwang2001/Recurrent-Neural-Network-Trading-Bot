from msilib import sequence
from multiprocessing.reduction import sendfds
from ob import *
from datetime import datetime
from time import *
from asyncio import *
import pandas as pd
import nest_asyncio
from ib_insync import *
nest_asyncio.apply()


class optionsbot(sendorder):
    def __init__(self):
        """
        Connect to IBKR api using port number 7497 (simulated trading) or 7496 (live trading)
        Used in conjunction with the RNN to predict prices
        """

        try:
            self.ib = IB()
            self.ib.connect('127.0.0.1', 7497, clientId=1)
        except:
            print("Error: Failed to Establish Remote Connection")
            print("IB connection estabilisehd?: ", self.ib.isConnected())
            exit()

        self.account = "DU5541128"
        self.contID = 76792991  # Tesla
        self.arcontract = Stock(
            symbol='TSLA', exchange='SMART', currency='USD')
        self.curcontract = Forex('EURUSD')
        self.contract4 = Future('ES', '20220708', 'GLOBEX')
        self.contract3 = Stock(
            symbol='TSLA', exchange='SMART', currency='USD')
        self.buyorder = MarketOrder("Buy", 1)
        self.sellorder = MarketOrder("Sell", 1)

    async def main(self):
        # _choice = int(
        # input("Enter '1'minute or '5'minute for time scalping interval: "))
        _choice = 1

        # while there isn't an active order, execute
        while True:
            clonecontracttask = asyncio.create_task(self.clone_contract())
            dailyprofitlosstasak = asyncio.create_task(
                self.daily_profit_loss())
            # self.on_ticker_update(contractid=self.curcontract)
            print("---Checking Entry Criteria---")
            _criteria = []
            # 1 minute time frame
            if _choice == 1:
                _list_time = ["0000", "0100", "0200", "0300",
                              "0400", "0500", "0600", "0700", "0800", "0900"]
                _criteria.extend(_list_time)
                _x = 900
                for i in range(51):
                    _x += 100
                    _criteria.append(str(_x))

            # 5 minute time frame
            elif _choice == 5:
                _list_time = ["0000", "0500"]
                _criteria.extend(_list_time)
                _x = 400
                for i in range(11):
                    _x += 500
                    _criteria.append(str(_x))
            else:
                print("---Please enter the number 1 or 5---")
                break
            # add entries to list (once list has 2 entires proceed)
            _time_list = []
            await clonecontracttask
            await clonecontracttask
            await dailyprofitlosstasak
            while len(_time_list) != 2:
                print(strftime("\n\t---%H:%M:%S---"))
                self.str_time = datetime.today().strftime(
                    '%Y%m%d ') + strftime("%H:%M:%S")

                # display _contract or underlying stock _price depending on open positions
                if len(self.ib.positions(account=self.account)) == 0:
                    # displays _contract 3 information
                    self.liveStream(self.contract3, 3, durationstring='360 S',
                                    barSizeSettings='1 secs')
                else:
                    print("---Position is being monitored---")
                    # if there is a live order it will display _contract information instead
                    print(self.trade_profit_loss())

                for i in _criteria:
                    _current_minute = str((strftime("%M%S""")))
                    _current_time = str((strftime("%H:%M:%S")))
                    if _current_minute in _criteria:
                        _time_list.append(_current_time)
                        print(_time_list, " has been added")
                        self.ib.sleep(1)
                        break
                    elif _current_minute not in _criteria:
                        self.ib.sleep(1)

            # retrieve data using the given times and construct a table
            print("--- Constructing Tables ---")
            _storage = []
            for i in range(2):
                _placeholder = ''.join(_time_list[i])
                self.str_time = datetime.today().strftime('%Y%m%d ') + _placeholder
                self.liveStream(self.contract3, display=2,
                                durationstring='360 S', barSizeSettings='1 min')
                # checking if current close is higher or lower than previous close
                if _choice == 5:
                    _df1 = str(self.dfg.iloc[0, 1])
                    _df2 = str(self.dfg.iloc[5, 4])
                elif _choice == 1:
                    _df1 = str(self.dfg.iloc[4, 1])
                    _df2 = str(self.dfg.iloc[5, 4])
                _storage.append(_df1)
                _storage.append(_df2)
            print("\nData Extracted: ", _storage)
            print("\n--- Analyzing Data ---")

            # if there is an active order place a trade on another _contract else place a normal trade
            if len(self.ib.positions(account=self.account)) == 0:
                ("No active trades currently, all positions closed")
                for i in range(1):
                    if _storage[1] > _storage[0] and _storage[3] > _storage[2]:
                        self.placeTrade("call", 1.05, 0.96)
                        print("Call Order Placed")

                    elif _storage[1] < _storage[0] and _storage[3] < _storage[2]:
                        if len(self.ib.positions(account=self.account)) == 0:
                            self.placeTrade("put", 1.05, 0.96)
                            print("Put order placed")

                    else:
                        print("---Criteria unsatisfactory, trying again---")
                        break
            else:
                print("Active trade in place, finding alternative order")
                self.placeTrade(
                    self.side(), 1.05, 0.96, 1)
                for confirm in self.ib.positions(account=self.account):
                    _side = confirm.contract.right
                print(f'---{_side} Order Placed---')

            print("---Generating Report---")
            self.comissions()
            self.active_trades()


if __name__ == "__main__":
    event_loop = asyncio.get_event_loop()
    ob = optionsbot()
    asyncio.run(ob.main())
    try:
        event_loop.run_forever()
    except KeyboardInterrupt:
        event_loop.close()
        print("interrupted")
    finally:
        exit()
