import time
from tkinter import NONE
from ib_insync.client import *
from ib_insync import *
from datetime import datetime
import asyncio
import nest_asyncio
nest_asyncio.apply()


class contractOrders:
    """
    Provides asynchronous functions using asyncio event looping
    Used in conjunction with the ib_insync library, in turn connected to interactive brokers API
    """

    def __init__(self):

        self.ib = IB()
        self.account = "DU5541128"
        self.contID = 76792991  # Tesla
        self.arcontract = Stock(
            symbol='TSLA', exchange='SMART', currency='USD')
        self.curcontract = Forex('EURUSD')
        self.contract4 = Future('ES', '20220708', 'GLOBEX')
        self.contract3 = Stock(symbol='TSLA', exchange='SMART', currency='USD')
        self.buyorder = MarketOrder("Buy", 1)
        self.sellorder = MarketOrder("Sell", 1)
        self._lastfillprice = []


class optionsChain(contractOrders):
    """Retrieves information from options chain"""
    async def clone_contract(self):
        """
        Creation of a clonecontract, purchase another order at a different strike if
        a trade is currently active
        """
        _st = time.time()
        await asyncio.sleep(0)
        for _position in self.ib.positions(self.account):
            if len(_position) >= 1:
                # copies the options side and buys 1 strike beyond current strike
                self._clonecontract = Option(
                    "TSLA", self.expiry(), self.strike() + 5, self.side(), 'SMART')
                return self._clonecontract
            else:
                pass
        _et = time.time()
        print(f'options_data execution time: {_et - _st}')

    def expiry(self):
        """Returns the most recent expiry data"""
        _optchain = self.optchain = self.ib.reqSecDefOptParams(
            underlyingSymbol="TSLA", underlyingSecType="STK", futFopExchange="", underlyingConId=self.contID)
        for categories in _optchain:
            _expiry = categories.expirations[0]
        return _expiry

    def strike(self):
        """Return position last strike price"""
        for _position in self.ib.positions(account=self.account):
            _strike = _position.contract.strike
        return _strike

    def side(self):
        """Return position call or put side of the contract"""
        for _position in self.ib.positions(account=self.account):
            _side = _position.contract.right
        return _side

    def contract(self):
        """Return the last used contractid and other fields from last trade"""
        for order in self.ib.trades():
            return order.contract


class stream(contractOrders):
    """security, options, market depth price streaming"""

    def loopstream(self):
        """updates price every second"""
        while True:
            self.liveStream(self.contract3, 3, durationstring='30 S',
                            barSizeSettings='1 secs')
            self.ib.sleep(10)

    def liveStream(self, contractid, display: int, durationstring='360 S', barSizeSettings='1 min', datafarm=False):
        """
        Returns live security or options price data
        args:
            contractid = which contract you want to use
            display: Choose which data type to return
                1: non dataframe bars, 2: return dataframe bars, 3: return security last close price +
                dataframe, 4: security last close price only, 5: option last close price + dataframe, 6: for datafarm usage only
            durationString: Time span of all the bars. Examples:
                '60 S', '30 D', '13 W', '6 M', '10 Y'.
            barSizeSettings: Time period of one bar. Must be one of:
                '1 secs', '5 secs', '10 secs' 15 secs', '30 secs',
                '1 min', '2 mins', '3 mins', '5 mins', '10 mins', '15 mins',
                '20 mins', '30 mins',
                '1 hour', '2 hours', '3 hours', '4 hours', '8 hours',
                '1 day', '1 week', '1 month'.
        """
        _bars = self.ib.reqHistoricalData(
            contract=contractid,
            endDateTime='',
            durationStr=durationstring,
            barSizeSetting=barSizeSettings,
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1,
            keepUpToDate=True)
        self.dfg = util.df(_bars)
        # for data in _bars:
        # sec_last_close = round(data.close, 1)
        sec_last_close = round(self.dfg.iloc[-1, 4], -1)
        opt_last_close = round(self.dfg.iloc[-1, 4], 2)
        # dictionary of print and return values
        _data_list = {1: [self.dfg.iloc[:6, :5], _bars],
                      2: ['', self.dfg.iloc[:6, :5]],
                      3: [self.dfg.iloc[-1, :5], sec_last_close],
                      4: ['', sec_last_close],
                      5: [self.dfg.iloc[-1, :5], opt_last_close],
                      6: [self.dfg.iloc[:, 0], self.dfg.iloc[:, 4]]}
        # print and return respective values
        if datafarm == True:
            _close = zip([time for time in self.dfg.iloc[:, 0]],
                         [close_price for close_price in self.dfg.iloc[:, 4]])

            _open = [open_price for open_price in self.dfg.iloc[:, 1]]
            _high = [high_price for high_price in self.dfg.iloc[:, 2]]
            _low = [low_price for low_price in self.dfg.iloc[:, 3]]
            _volume = [_volume for _volume in self.dfg.iloc[:, 5]]
            _price_list = [_close, _open, _high, _low, _volume]
            return _price_list
        else:
            print(_data_list[display][0])
            return _data_list[display][1]

    def volatility_data(self):
        _bars = self.ib.reqHistoricalData(
            contract=self.contract3,
            endDateTime='',
            durationStr="60 D",
            barSizeSetting="1 day",
            whatToShow='HISTORICAL_VOLATILITY',
            useRTH=True,
            formatDate=1,
            keepUpToDate=False)
        df = util.df(_bars)
        return df

    def on_ticker_update(self, contractid):
        # Market Detph Streaming
        self.ib.qualifyContracts(contractid)
        self.ib.reqTickers(self.contract3)
        self.ib.reqMarketDataType(1)
        _ticker = self.ib.reqMktDepth(contractid, isSmartDepth=True)
        _ticker.marketPrice()
        bids = [d._price for d in _ticker.domTicks]
        asks = [d._price for d in _ticker.domTicks]

        if len(bids) >= 1:
            print(bids, asks)
        else:
            pass


class statistics(optionsChain, stream):
    """display important stats such as profit and loss, active tradaes and comissions"""
    async def daily_profit_loss(self):
        """prints account pnl for the day"""
        _st = time.time()
        self.ib.reqPnL(account=self.account, modelCode='')
        await asyncio.sleep(1)
        _pnl = self.ib.pnl()
        """self.ib.reqPnLSingle(account=self.account,
                             conId=self.contID, modelCode='')
        await asyncio.sleep(1)
        _singlepnl = self.ib.pnlSingle()"""
        print(_pnl, "\n")
        self.ib.cancelPnL(account=self.account,
                          modelCode='')
        _et = time.time()
        print(f'options_data execution time: {_et - _st}')

    def trade_profit_loss(self):
        """shows pnl on active trade"""
        for order in self.ib.trades():
            self._fields_list = Option(order.contract.symbol, order.contract.lastTradeDateOrContractMonth,
                                       order.contract.strike, order.contract.right, order.contract.exchange)
            break
        _lastclose = self.liveStream(self._fields_list, 5, durationstring='30 S',
                                     barSizeSettings='1 secs')
        co = contractOrders()
        if _lastclose is None or len(co._lastfillprice) == 0:
            pass
        else:
            if self.side() == 'C':
                print(
                    f'(Active Trade in place, Call order at strike {self.strike()})')
            else:
                print(
                    f'(Active Trade in place, Put order at strike {self.strike()})')
            return f"(Profit and loss: {(_lastclose - co._lastfillprice[-1])*100} dollars.)\n"

    def active_trades(self) -> None:
        print("Open Orders: ", self.ib.openTrades())

    def comissions(self):
        comissions = ("Total Comissions Paid for the day: ", sum(
            fill.commissionReport.commission for fill in self.ib.fills()))
        print(comissions)


class sendorder(statistics, IB):

    """place trades and configure order types"""

    def placeTrade(self, option, takeprofitmult=float,
                   stoplossmult=float, specialorder='') -> None:
        """
        args:
            option: Type "call" or "put"

            takeprofitmult = Type in percentage of parent order in decimals ex. 1.05 = Sell Limit

            stoplossmult = Type in percentage of parent order in decimals ex. 0.97 = Stop trigger

            clonecontract = used to duplicate an existing order

                            if specialorder parameter == 1
                                return clonecontract

                            syntax: parameter must have option =''
                            and specialorder=1 to use clonecontract

        Attributes:
            clonecontract = contract that mimcs the previous trade performed but buys a different strike, this can be used to
                            prevent duplicate order errors
            _callcontract = contract strike is 2 strikes above current price
            _putcontract = contract strike is 2 strikes below current price
            _callchart = dataframe call prices
            _putchart = dataframe put prices

        Return None
        """

        _st = time.time()
        _lastclose = self.liveStream(self.contract3, 4, durationstring='30 S',
                                     barSizeSettings='1 secs')
        _callcontract = Option("TSLA", self.expiry(),
                               _lastclose + 10, 'C', 'SMART')
        _putcontract = Option(
            "TSLA", self.expiry(), _lastclose - 10, 'P', 'SMART')

        _callchart = self.liveStream(_callcontract, display=5,
                                     durationstring='360', barSizeSettings='1 min')

        _putchart = self.liveStream(_putcontract, display=5,
                                    durationstring='360', barSizeSettings='1 min')

        self.takeprofit, self.stoploss = takeprofitmult, stoplossmult

        # place normal call or put order
        if option == "call" or option == "C":
            _chart_type, _contract = _callchart, _callcontract

        elif option == "put" or option == "P":
            _chart_type, _contract = _putchart, _putcontract

        # place special order
        elif specialorder == 1:
            _contract = self._clonecontract

        _bracket = self._customBracketOrder(
            "BUY", 1, takeProfitPrice=round(_chart_type * self.takeprofit, 2), stopLossPrice=round(_chart_type * self.stoploss, 2))

        # transmits parent and child orders
        for entire in _bracket:
            self.order = self.ib.placeOrder(
                _contract, entire)

        for self.log in self.order.log:
            print("\n", self.log.status) if self.log.status not in OrderStatus.DoneStates else print(
                "Finished")
        self.ib.sleep(10)
        _fill = self.order.orderStatus.lastFillPrice
        print(_fill)
        co = contractOrders()
        co._lastfillprice.append(_fill)
        _et = time.time()
        print(f"\nPlaceTrade Function Execution Time: {_et - _st} seconds.\n")

    def _customBracketOrder(
            self, action: str, quantity: float,
            takeProfitPrice: float,
            stopLossPrice: float, **kwargs) -> BracketOrder:
        """
        Create a limit order that is bracketed by a take-profit order and
        a stop-loss order. Submit the bracket like:

        .. code-block:: python

            for o in bracket:
                ib.placeOrder(contract, o)

        https://interactivebrokers.github.io/tws-api/bracket_order.html

        Args:
            action: 'BUY' or 'SELL'.
            quantity: Size of order.
            limitPrice: Limit price of entry order.
            takeProfitPrice: Limit price of profit order.
            stopLossPrice: Stop price of loss order.
        """
        assert action in ('BUY', 'SELL')
        reverseAction = 'BUY' if action == 'SELL' else 'SELL'
        parent = MarketOrder(
            action, quantity,
            orderId=self.ib.client.getReqId(),
            transmit=False,
            **kwargs)
        takeProfit = LimitOrder(
            reverseAction, quantity, takeProfitPrice,
            orderId=self.ib.client.getReqId(),
            transmit=False,
            parentId=parent.orderId,
            **kwargs)
        stopLoss = StopOrder(
            reverseAction, quantity, stopLossPrice,
            orderId=self.ib.client.getReqId(),
            transmit=True,
            parentId=parent.orderId,
            **kwargs)
        return BracketOrder(parent, takeProfit, stopLoss)

    if __name__ == "__main__":
        ib = IB()
        ib.connect('127.0.0.1', 7497, clientId=1)
        st = statistics()
        st.trade_profit_loss()
