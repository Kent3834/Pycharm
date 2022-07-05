from binance.client import Client
btc_price = Client.get_symbol_ticker(self, symbol="BTCUSDT")
print(btc_price)