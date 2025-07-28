import time
import requests
import hashlib
import hmac
import base64
from urllib.parse import urlencode

from systems.scripts.kraken_auth import load_kraken_keys
from systems.utils.resolve_symbol import resolve_symbol
from systems.utils.logger import addlog

KRAKEN_ORDER_TIMEOUT = 6
SLIPPAGE_STEPS = [0.0, 0.002, 0.004, 0.007, 0.01]

def _kraken_request(endpoint: str, data: dict, api_key: str, api_secret: str) -> dict:
    url_path = f"/0/private/{endpoint}"
    url = f"https://api.kraken.com{url_path}"

    nonce = str(int(time.time() * 1000))
    data["nonce"] = nonce

    post_data = urlencode(data)
    encoded = (nonce + post_data).encode()
    message = url_path.encode() + hashlib.sha256(encoded).digest()

    signature = hmac.new(base64.b64decode(api_secret), message, hashlib.sha512)
    sig_digest = base64.b64encode(signature.digest())

    headers = {
        "API-Key": api_key,
        "API-Sign": sig_digest.decode()
    }

    resp = requests.post(url, headers=headers, data=data, timeout=10)
    result = resp.json()
    if "error" in result and result["error"]:
        raise Exception(f"Kraken API error: {result['error']}")
    return result

def get_kraken_balance(verbose: int = 0) -> dict:
    api_key, api_secret = load_kraken_keys()
    result = _kraken_request("Balance", {}, api_key, api_secret).get("result", {})
    addlog(
        f"[INFO] Kraken balance fetched: {result}",
        verbose_int=2,
        verbose_state=verbose,
    )
    return {k: float(v) for k, v in result.items()}


def get_available_fiat_balance(exchange, currency: str = "USD") -> float:
    """Return available fiat balance from a CCXT exchange object."""
    try:
        balance = exchange.fetch_free_balance()
    except Exception:
        return 0.0
    return float(balance.get(currency, 0.0))

def buy_order(symbol: str, usd_amount: float, verbose: int = 0) -> dict:
    api_key, api_secret = load_kraken_keys()
    symbols = resolve_symbol(symbol)
    pair_code = symbols["kraken"]

    # Check balance
    balance = get_kraken_balance(verbose)
    available_usd = balance.get("ZUSD", 0.0)
    if available_usd < usd_amount:
        addlog(
            f"[ABORT] Not enough USDT to buy: ${available_usd:.2f} available, need ${usd_amount:.2f}",
            verbose_int=1,
            verbose_state=verbose,
        )
        return {}

    for slippage in SLIPPAGE_STEPS:
        price_resp = requests.get(
            f"https://api.kraken.com/0/public/Ticker?pair={pair_code}"
        ).json()
        ticker_result = price_resp.get("result", {})
        if not ticker_result:
            addlog(
                "[ERROR] Invalid ticker response: missing result",
                verbose_int=1,
                verbose_state=verbose,
            )
            continue
        ticker_key = next(iter(ticker_result))
        ticker_data = ticker_result.get(ticker_key, {})
        close = ticker_data.get("c")
        if not close:
            addlog(
                "[ERROR] Invalid ticker response: missing close price",
                verbose_int=1,
                verbose_state=verbose,
            )
            continue
        price = float(close[0])
        adjusted_price = price * (1 + slippage)
        coin_amount = round(usd_amount / adjusted_price, 8)

        addlog(
            f"\nTrying buy with slippage {slippage*100:.2f}% → volume {coin_amount:.6f}",
            verbose_int=1,
            verbose_state=verbose,
        )

        order_resp = _kraken_request("AddOrder", {
            "pair": pair_code,
            "type": "buy",
            "ordertype": "market",
            "volume": coin_amount,
            "trades": True
        }, api_key, api_secret)

        txid = order_resp["result"]["txid"][0]
        addlog(f"Order placed: {txid}", verbose_int=1, verbose_state=verbose)

        start = time.time()
        while time.time() - start < KRAKEN_ORDER_TIMEOUT:
            trades_resp = _kraken_request("TradesHistory", {}, api_key, api_secret)
            trades = trades_resp["result"]["trades"]
            for tid, trade in trades.items():
                if trade["ordertxid"] == txid:
                    addlog("Trade found in history", verbose_int=1, verbose_state=verbose)
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

        addlog("Slippage level failed, trying next...", verbose_int=1, verbose_state=verbose)

    raise Exception("Buy order failed — no fill found within timeout.")

def sell_order(symbol: str, usd_amount: float, verbose: int = 0) -> dict:
    api_key, api_secret = load_kraken_keys()
    symbols = resolve_symbol(symbol)
    pair_code = symbols["kraken"]

    price_resp = requests.get(
        f"https://api.kraken.com/0/public/Ticker?pair={pair_code}"
    ).json()
    ticker_result = price_resp.get("result", {})
    if not ticker_result:
        raise Exception("Invalid ticker response: missing result")
    ticker_key = next(iter(ticker_result))
    ticker_data = ticker_result.get(ticker_key, {})
    close = ticker_data.get("c")
    if not close:
        raise Exception("Invalid ticker response: missing close price")
    price = float(close[0])
    coin_amount = round(usd_amount / price, 8)

    order_resp = _kraken_request("AddOrder", {
        "pair": pair_code,
        "type": "sell",
        "ordertype": "market",
        "volume": coin_amount,
        "trades": True
    }, api_key, api_secret)

    txid = order_resp["result"]["txid"][0]
    addlog(f"Sell Order placed: {txid}", verbose_int=1, verbose_state=verbose)

    start = time.time()
    while time.time() - start < KRAKEN_ORDER_TIMEOUT:
        trades_resp = _kraken_request("TradesHistory", {}, api_key, api_secret)
        trades = trades_resp["result"]["trades"]
        for tid, trade in trades.items():
            if trade["ordertxid"] == txid:
                addlog("Sell trade found in history", verbose_int=1, verbose_state=verbose)
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
