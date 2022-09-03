from threading import activeCount
import time
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


class statistics(contractOrders):
    """display important stats such as profit and loss, active tradaes and comissions"""
    async def daily_profit_loss(self):
        """prints account pnl for the day"""
        _st = time.time()
        await asyncio.sleep(0)
        self.ib.reqPnL(account=self.account, modelCode='')
        _pnl = ("PnL: ", self.ib.pnl())
        self.ib.reqPnLSingle(account=self.account,
                             conId=self.contID, modelCode='')
        singlepnl = ("Position Pnl: ", self.ib.pnlSingle())
        self.ib.cancelPnLSingle(account=self.account,
                                conId=self.contID, modelCode='')
        print(_pnl)
        _et = time.time()
        print(f'options_data execution time: {_et - _st}')

    def trade_profit_loss(self):
        """prints pnl per trade"""
        self.order.orderStatus.lastFillPrice
        pass

    def active_trades(self) -> None:
        print("Open Orders: ", self.ib.openOrders())

    def comissions(self):
        comissions = ("Total Comissions Paid for the day: ", sum(
            fill.commissionReport.commission for fill in self.ib.fills()))
        print(comissions)


class stream(contractOrders):
    """security, options, market depth price streaming"""

    def liveStream(self, contractid, display: int, durationstring='360 S', barSizeSettings='1 min'):
        """
        Returns live security or options price data
        args:
            contractid = which contract you want to use
            display: Choose which data type to return
                1: non dataframe bars, 2: return dataframe bars, 3: return last close
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
            whatToShow='ASK',
            useRTH=False,
            formatDate=1,
            keepUpToDate=True)
        self.dfg = util.df(_bars)
        for data in _bars:
            _lastclose = round(data.close, -1)
        # dictionary of print and return values
        _data_list = {1: [self.dfg.iloc[:6, :5], _bars], 2: [
            '', self.dfg.iloc[:6, :5]], 3: [self.dfg.iloc[-1, :5], _lastclose]}
        # print and return respective values
        print(_data_list[display][0])
        return _data_list[display][1]

    def on_ticker_update(self, contract):
        # Market Detph Streaming
        self.ib.qualifyContracts(contract)
        self.ib.reqTickers(self.contract3)
        self.ib.reqMarketDataType(1)
        _ticker = self.ib.reqMktDepth(contract, isSmartDepth=True)
        print(
            [d._price for d in _ticker.domTicks],
            [d._price for d in _ticker.domTicks])


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

    async def expiry(self):
        """Returns the most recent expiry data"""
        await asyncio.sleep(0)
        _optchain = self.ib.reqSecDefOptParams()
        for categories in _optchain:
            _expiry = categories.expirations[0]
        return _expiry

    async def strike(self):
        """Return an alternative strike price"""
        await asyncio.sleep(0)
        for _position in self.ib.positions(account=self.account):
            _strike = _position.contract.strike
        return _strike

    async def side(self):
        """Return call or put side of the contract"""
        await asyncio.sleep(0)
        for _position in self.ib.positions(account=self.account):
            _side = _position.contract.right
        return _side


class sendorder(optionsChain, stream, Client):

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
        _callcontract = Option("TSLA", self.expiry(),
                               self.strike() + 10, 'C', 'SMART')
        _putcontract = Option(
            "TSLA", self.expiry(), self.strike() - 10, 'P', 'SMART')

        _callchart = self.liveStream(self.contract3, display=2,
                                     durationstring='360', barSizeSettings='1 min')

        _putchart = self.liveStream(self.contract3, display=2,
                                    durationstring='360', barSizeSettings='1 min')

        self.takeprofit, self.stoploss = takeprofitmult, stoplossmult

        # place normal call or put order
        if option == "call":
            _chart_type, _contract = _callchart, _callcontract
        elif option == "put":
            _chart_type, _contract = _putchart, _putcontract
        # place special order
        elif option == '' and specialorder == 1:
            _contract = self.clonecontract
            if self.side == 'C':
                _chart_type = _callchart
            elif self.side == 'P':
                _chart_type - _putchart
        else:
            print("Invalid Entry")

        _price = float(_chart_type.iloc[5, 4])
        bracket = self._customBracketOrder(
            "BUY", 1, takeProfitPrice=round(_price * self.takeprofit, 2), stopLossPrice=round(_price * self.stoploss, 2))
        # transmits parent and child orders
        for entire in bracket:
            self.order = self.ib.placeOrder(
                _contract, entire)
        for self.log in self.order.log:
            print(self.log.status) if self.log.status not in OrderStatus.DoneStates else print(
                "Finished")
        _et = time.time()
        print(f"\nPlaceTrade Function Execution Time: {_et - _st} seconds.\n")

    def _custom_bracket_order(self, action: str, quantity: float,
                              takeProfitPrice: float, stopLossPrice: float, **kwargs) -> BracketOrder:
        """alternative version to BracketOrder, uses Market Order for parent instead of limit order"""
        self.wrapper = Wrapper(self)
        self.client = Client(self.wrapper)
        assert action in ('BUY', 'SELL')
        reverseAction = 'BUY' if action == 'SELL' else 'SELL'

        parent = MarketOrder(
            action, quantity,
            orderId=self.client.getReqId(),
            transmit=False,
            **kwargs)
        takeProfit = LimitOrder(
            reverseAction, quantity, takeProfitPrice,
            orderId=self.client.getReqId(),
            transmit=False,
            parentId=parent.orderId,
            **kwargs)
        stopLoss = StopOrder(
            reverseAction, quantity, stopLossPrice,
            orderId=self.client.getReqId(),
            # When stoploss is queued, transmit is set to True to send all 3 orders to the server
            transmit=True,
            parentId=parent.orderId,
            **kwargs)
        return BracketOrder(parent, takeProfit, stopLoss)
