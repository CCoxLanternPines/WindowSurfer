import time
from systems.scripts.kraken_auth import load_kraken_keys
from systems.utils.resolve_symbol import resolve_symbol
from systems.scripts.kraken_utils import _kraken_request
import requests

KRAKEN_ORDER_TIMEOUT = 6  # seconds to wait for fill confirmation
SLIPPAGE_STEPS = [0.0, 0.002, 0.004, 0.007, 0.01]  # 0% to 1%

def buy_order(symbol: str, usd_amount: float, verbose: int = 0) -> dict:
    api_key, api_secret = load_kraken_keys()
    symbols = resolve_symbol(symbol)
    pair_code = symbols["kraken"]

    for slippage in SLIPPAGE_STEPS:

        price_resp = requests.get(f"https://api.kraken.com/0/public/Ticker?pair={pair_code}").json()
        price = float(price_resp["result"][pair_code]["c"][0])
        adjusted_price = price * (1 + slippage)
        coin_amount = round(usd_amount / adjusted_price, 8)

        if verbose >= 1:
            print(f"\nTrying buy with slippage {slippage*100:.2f}% → volume {coin_amount:.6f}")

        order_resp = _kraken_request("AddOrder", {
            "pair": pair_code,
            "type": "buy",
            "ordertype": "market",
            "volume": coin_amount
        }, api_key, api_secret)

        txid = order_resp["result"]["txid"][0]
        if verbose >= 1:
            print(f"Order placed: {txid}")

        start = time.time()
        while time.time() - start < KRAKEN_ORDER_TIMEOUT:
            trades_resp = _kraken_request("TradesHistory", {}, api_key, api_secret)
            trades = trades_resp["result"]["trades"]
            for tid, trade in trades.items():
                if trade["ordertxid"] == txid:
                    if verbose >= 1:
                        print("Trade found in history")
                    return {
                        "kraken_txid": txid,
                        "symbol": symbol,
                        "price": float(trade["price"]),
                        "volume": float(trade["vol"]),
                        "cost": float(trade["cost"]),
                        "fee": float(trade["fee"]),
                        "timestamp": int(trade["time"])
                    }
            time.sleep(0.6)

        if verbose >= 1:
            print("Slippage level failed, trying next...")

    raise Exception("Buy order failed — no fill found within timeout.")

def sell_order(symbol: str, usd_amount: float, verbose: int = 0) -> dict:
    api_key, api_secret = load_kraken_keys()
    symbols = resolve_symbol(symbol)
    pair_code = symbols["kraken"]
    price_resp = requests.get(f"https://api.kraken.com/0/public/Ticker?pair={pair_code}").json()
    price = float(price_resp["result"][pair_code]["c"][0])
    coin_amount = round(usd_amount / price, 8)

    order_resp = _kraken_request("AddOrder", {
        "pair": pair_code,
        "type": "sell",
        "ordertype": "market",
        "volume": coin_amount
    }, api_key, api_secret)

    txid = order_resp["result"]["txid"][0]
    if verbose >= 1:
        print(f"Sell Order placed: {txid}")

    start = time.time()
    while time.time() - start < KRAKEN_ORDER_TIMEOUT:
        trades_resp = _kraken_request("TradesHistory", {}, api_key, api_secret)
        trades = trades_resp["result"]["trades"]
        for tid, trade in trades.items():
            if trade["ordertxid"] == txid:
                if verbose >= 1:
                    print("Sell trade found in history")
                return {
                    "kraken_txid": txid,
                    "symbol": symbol,
                    "price": float(trade["price"]),
                    "volume": float(trade["vol"]),
                    "cost": float(trade["cost"]),
                    "fee": float(trade["fee"]),
                    "timestamp": int(trade["time"])
                }
        time.sleep(0.6)

    raise Exception("Sell order failed — no fill found within timeout.")
