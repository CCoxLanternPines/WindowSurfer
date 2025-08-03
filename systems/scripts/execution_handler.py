import time
import requests
import hashlib
import hmac
import base64
from urllib.parse import urlencode

from systems.scripts.kraken_auth import load_kraken_keys
from systems.utils.addlog import addlog
from systems.scripts.kraken_utils import ensure_snapshot, get_live_price  # use shared util now

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


def get_available_fiat_balance(exchange, currency: str = "USD") -> float:
    try:
        balance = exchange.fetch_free_balance()
    except Exception:
        return 0.0
    return float(balance.get(currency, 0.0))

def buy_order(
    pair_code: str,
    fiat_symbol: str,
    usd_amount: float,
    ledger_name: str,
    wallet_code: str,
    verbose: int = 0,
) -> dict:
    api_key, api_secret = load_kraken_keys()

    snapshot = ensure_snapshot(ledger_name)
    if not snapshot:
        addlog(
            "[ABORT] Kraken snapshot unavailable — cannot place buy order",
            verbose_int=1,
            verbose_state=verbose,
        )
        return {}
    balance = snapshot.get("balance", {})
    available_usd = float(balance.get(wallet_code, 0.0))
    if available_usd < usd_amount:
        addlog(
            f"[ABORT] Not enough {fiat_symbol} to buy: ${available_usd:.2f} available, need ${usd_amount:.2f}",
            verbose_int=1,
            verbose_state=verbose,
        )
        return {}

    addlog(
        f"[BUY ATTEMPT] {fiat_symbol} available: ${available_usd:.2f}, attempting to buy ${usd_amount:.2f}",
        verbose_int=3,
        verbose_state=verbose,
    )

    for slippage in SLIPPAGE_STEPS:
        price_resp = requests.get(f"https://api.kraken.com/0/public/Ticker?pair={pair_code}").json()
        ticker_result = price_resp.get("result", {})
        if not ticker_result:
            addlog(
                "[ERROR] Invalid ticker response: missing result",
                verbose_int=2,
                verbose_state=verbose,
            )
            continue
        ticker_key = next(iter(ticker_result))
        ticker_data = ticker_result.get(ticker_key, {})
        close = ticker_data.get("c")
        if not close:
            addlog(
                "[ERROR] Invalid ticker response: missing close price",
                verbose_int=2,
                verbose_state=verbose,
            )
            continue

        price = float(close[0])
        adjusted_price = price * (1 + slippage)
        coin_amount = round(usd_amount / adjusted_price, 8)
        usd_equiv = coin_amount * adjusted_price

        addlog(
            f"\nTrying buy with slippage {slippage*100:.2f}% → ${usd_equiv:.2f}",
            verbose_int=3,
            verbose_state=verbose,
        )

        order_resp = _kraken_request(
            "AddOrder",
            {
                "pair": pair_code,
                "type": "buy",
                "ordertype": "market",
                "volume": coin_amount,
                "trades": True,
            },
            api_key,
            api_secret,
        )

        txid = order_resp["result"]["txid"][0]
        addlog(f"Order placed: {txid}", verbose_int=1, verbose_state=verbose)

        snapshot = ensure_snapshot(ledger_name)
        trades = snapshot.get("trades", {}) if snapshot else {}
        for tid, trade in trades.items():
            if trade.get("ordertxid") == txid:
                addlog(
                    "Trade found in snapshot history", verbose_int=2, verbose_state=verbose
                )
                return {
                    "kraken_txid": txid,
                    "symbol": pair_code,
                    "price": float(trade["price"]),
                    "volume": float(trade["vol"]),
                    "cost": float(trade["cost"]),
                    "fee": float(trade["fee"]),
                    "timestamp": int(trade["time"]),
                }
        addlog("Slippage level failed, trying next...", verbose_int=3, verbose_state=verbose)

    raise Exception("Buy order failed — no fill found in snapshot.")

def sell_order(
    pair_code: str, fiat_symbol: str, usd_amount: float, ledger_name: str, verbose: int = 0
) -> dict:
    api_key, api_secret = load_kraken_keys()

    if not ensure_snapshot(ledger_name):
        addlog(
            "[ABORT] Kraken snapshot unavailable — cannot place sell order",
            verbose_int=1,
            verbose_state=verbose,
        )
        return {}

    price_resp = requests.get(f"https://api.kraken.com/0/public/Ticker?pair={pair_code}").json()
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
    usd_amount = coin_amount * price
    addlog(
        f"[SELL ATTEMPT] Attempting to sell ${usd_amount:.2f} worth of {pair_code}",
        verbose_int=3,
        verbose_state=verbose,
    )

    order_resp = _kraken_request(
        "AddOrder",
        {
            "pair": pair_code,
            "type": "sell",
            "ordertype": "market",
            "volume": coin_amount,
            "trades": True,
        },
        api_key,
        api_secret,
    )

    txid = order_resp["result"]["txid"][0]
    addlog(f"Sell Order placed: {txid}", verbose_int=1, verbose_state=verbose)

    snapshot = ensure_snapshot(ledger_name)
    trades = snapshot.get("trades", {}) if snapshot else {}
    for tid, trade in trades.items():
        if trade.get("ordertxid") == txid:
            addlog(
                "Sell trade found in snapshot history", verbose_int=2, verbose_state=verbose
            )
            return {
                "kraken_txid": txid,
                "symbol": pair_code,
                "price": float(trade["price"]),
                "volume": float(trade["vol"]),
                "cost": float(trade["cost"]),
                "fee": float(trade["fee"]),
                "timestamp": int(trade["time"]),
            }

    raise Exception("Sell order failed — no fill found in snapshot.")


def execute_buy(
    client,
    *,
    symbol: str,
    fiat_code: str,
    price: float,
    amount_usd: float,
    ledger_name: str,
    wallet_code: str,
    verbose: int = 0,
) -> dict:
    """Place a real buy order and normalise the result structure.

    Parameters are kept for API compatibility; ``client`` and ``price`` are
    currently unused as ``buy_order`` pulls pricing from Kraken directly.
    """

    fills = buy_order(symbol, fiat_code, amount_usd, ledger_name, wallet_code, verbose)
    if not fills:
        return {}
    return {
        "filled_amount": fills.get("volume", 0.0),
        "avg_price": fills.get("price", 0.0),
        "timestamp": fills.get("timestamp"),
    }


def execute_sell(
    client,
    *,
    symbol: str,
    coin_amount: float,
    fiat_code: str | None = None,
    price: float | None = None,
    ledger_name: str,
    verbose: int = 0,
) -> dict:
    """Place a real sell order and normalise the result structure.

    ``fiat_code`` defaults to ``ZUSD`` when not provided. ``price`` is optional
    and, if absent, the current live price is fetched to estimate USD notional.
    """

    fiat = fiat_code or "ZUSD"
    sell_price = price if price is not None else get_live_price(symbol)
    usd_amount = coin_amount * sell_price
    fills = sell_order(symbol, fiat, usd_amount, ledger_name, verbose)
    return {
        "filled_amount": fills.get("volume", 0.0),
        "avg_price": fills.get("price", 0.0),
        "timestamp": fills.get("timestamp"),
    }
