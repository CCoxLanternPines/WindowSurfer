import time
import requests
import hashlib
import hmac
import base64
import json
from urllib.parse import urlencode

from systems.scripts.kraken_auth import load_kraken_keys
from systems.utils.addlog import addlog, send_telegram_message
from systems.utils.path import find_project_root


def fetch_price_data(symbol: str) -> dict:
    resp = requests.get(
        f"https://api.kraken.com/0/public/Ticker?pair={symbol}", timeout=10
    )
    result = resp.json().get("result", {})
    return next(iter(result.values()), {})


def now_utc_timestamp() -> int:
    return int(time.time())


def load_snapshot(tag: str) -> dict | None:
    root = find_project_root()
    snap_path = root / "data" / "snapshots" / f"{tag}.json"
    if not snap_path.exists():
        return None
    try:
        with snap_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_snapshot(tag: str, snapshot: dict) -> None:
    root = find_project_root()
    snap_dir = root / "data" / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    with (snap_dir / f"{tag}.json").open("w", encoding="utf-8") as f:
        json.dump(snapshot, f)


def fetch_snapshot_from_kraken(tag: str) -> dict:
    api_key, api_secret = load_kraken_keys()
    balance_resp = _kraken_request("Balance", {}, api_key, api_secret).get("result", {})
    trades_resp = _kraken_request(
        "TradesHistory",
        {"type": "all", "trades": True},
        api_key,
        api_secret,
    ).get("result", {})
    return {
        "last_updated": now_utc_timestamp(),
        "balance": balance_resp,
        "trades": trades_resp.get("trades", trades_resp),
    }


def load_or_fetch_snapshot(tag: str) -> dict:
    snapshot = load_snapshot(tag)
    if snapshot is None or snapshot.get("last_updated", 0) < now_utc_timestamp() - 60:
        snapshot = fetch_snapshot_from_kraken(tag)
        save_snapshot(tag, snapshot)
    return snapshot


def _get_snapshot(tag: str, api_key: str, api_secret: str) -> dict:
    return load_or_fetch_snapshot(tag)

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

    snapshot = _get_snapshot(ledger_name, api_key, api_secret)
    balance = snapshot.get("balance", {})
    available_usd = float(balance.get(fiat_symbol, 0.0))

    addlog(f"[DEBUG] Balance snapshot: {balance}", verbose_int=1, verbose_state=verbose)
    addlog(f"[DEBUG] Using fiat_symbol = {fiat_symbol}", verbose_int=1, verbose_state=verbose)
    addlog(
        f"[DEBUG] available_usd = {available_usd} | trying to spend = {usd_amount}",
        verbose_int=1,
        verbose_state=verbose,
    )

    if available_usd < usd_amount:
        addlog(
            f"[SKIP] Not enough {fiat_symbol} to buy: ${available_usd:.2f} available, need ${usd_amount:.2f}",
            verbose_int=1,
            verbose_state=verbose,
        )
        return {}

    addlog(
        f"[BUY ATTEMPT] {fiat_symbol} available: ${available_usd:.2f}, attempting to buy ${usd_amount:.2f}",
        verbose_int=3,
        verbose_state=verbose,
    )

    price_data = fetch_price_data(pair_code)
    price = float(price_data.get("c", [0])[0])
    coin_amount = round(usd_amount / price, 8)

    try:
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
    except Exception as e:
        err_msg = str(e)
        if "EOrder:Insufficient funds" in err_msg:
            addlog(
                "[SKIP] Kraken rejected order — insufficient funds",
                verbose_int=1,
                verbose_state=verbose,
            )
            send_telegram_message(
                f"❗ Kraken rejected order due to insufficient funds on {ledger_name}."
            )
            return {}
        raise

    txid_list = order_resp.get("result", {}).get("txid")
    txid = txid_list[0] if txid_list else None
    if not txid:
        raise Exception("Buy order failed — no txid returned")

    return {
        "txid": txid,
        "volume": coin_amount,
        "price": price,
        "filled_amount": coin_amount,
        "avg_price": price,
        "timestamp": now_utc_timestamp(),
    }

def sell_order(
    pair_code: str, fiat_symbol: str, usd_amount: float, ledger_name: str, verbose: int = 0
) -> dict:
    api_key, api_secret = load_kraken_keys()

    snapshot = _get_snapshot(ledger_name, api_key, api_secret)

    price_data = fetch_price_data(pair_code)
    price = float(price_data.get("c", [0])[0])
    coin_amount = round(usd_amount / price, 8)
    usd_amount = coin_amount * price
    addlog(
        f"[SELL ATTEMPT] Attempting to sell ${usd_amount:.2f} worth of {pair_code}",
        verbose_int=3,
        verbose_state=verbose,
    )

    try:
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
    except Exception as e:
        err_msg = str(e)
        if "EOrder:Insufficient funds" in err_msg:
            addlog(
                "[SKIP] Kraken rejected order — insufficient funds",
                verbose_int=1,
                verbose_state=verbose,
            )
            send_telegram_message(
                f"❗ Kraken rejected order due to insufficient funds on {ledger_name}."
            )
            return {}
        raise

    txid_list = order_resp.get("result", {}).get("txid")
    txid = txid_list[0] if txid_list else None
    if not txid:
        raise Exception("Sell order failed — no txid returned")

    return {
        "txid": txid,
        "volume": coin_amount,
        "price": price,
        "filled_amount": coin_amount,
        "avg_price": price,
        "timestamp": now_utc_timestamp(),
    }


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

    result = buy_order(
        symbol, fiat_code, amount_usd, ledger_name, wallet_code, verbose
    )
    if (
        not result
        or result.get("filled_amount", 0) <= 0
        or result.get("price", 0) <= 0
        or not result.get("txid")
        or not result.get("timestamp")
    ):
        addlog(
            f"[SKIP] Trade result invalid — not logging to ledger for {ledger_name}",
            verbose_int=1,
            verbose_state=verbose,
        )
        send_telegram_message(
            f"⚠️ Skipped logging invalid trade for {ledger_name}: empty or failed result."
        )
        return
    return {
        "filled_amount": result.get("filled_amount", 0.0),
        "avg_price": result.get("price", 0.0),
        "timestamp": result.get("timestamp"),
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
    result = sell_order(symbol, fiat, usd_amount, ledger_name, verbose)
    if (
        not result
        or result.get("filled_amount", 0) <= 0
        or result.get("price", 0) <= 0
        or not result.get("txid")
        or not result.get("timestamp")
    ):
        addlog(
            f"[SKIP] Trade result invalid — not logging to ledger for {ledger_name}",
            verbose_int=1,
            verbose_state=verbose,
        )
        send_telegram_message(
            f"⚠️ Skipped logging invalid trade for {ledger_name}: empty or failed result."
        )
        return
    return {
        "filled_amount": result.get("filled_amount", 0.0),
        "avg_price": result.get("price", 0.0),
        "timestamp": result.get("timestamp"),
    }
