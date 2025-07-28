import time
import requests
import hashlib
import hmac
import base64
from urllib.parse import urlencode

from systems.scripts.kraken_auth import load_kraken_keys
from systems.utils.resolve_symbol import resolve_symbol

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
    if verbose >= 2:
        print("[INFO] Kraken balance fetched:", result)
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
        if verbose >= 1:
            print(f"[ABORT] Not enough USDT to buy: ${available_usd:.2f} available, need ${usd_amount:.2f}")
        return {}

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
            "volume": coin_amount,
            "trades": True
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
        "volume": coin_amount,
        "trades": True
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
