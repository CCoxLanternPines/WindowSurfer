import time
import requests
import hashlib
import hmac
import base64
from urllib.parse import urlencode

from systems.scripts.kraken_utils import load_kraken_keys
from systems.scripts.kraken_utils import get_live_price
from systems.utils.addlog import addlog, send_telegram_message
from systems.utils.resolve_symbol import split_tag
from systems.utils.quote_norm import norm_quote
from systems.scripts.trade_apply import apply_buy


def fetch_price_data(symbol: str) -> dict:
    resp = requests.get(
        f"https://api.kraken.com/0/public/Ticker?pair={symbol}", timeout=10
    )
    result = resp.json().get("result", {})
    return next(iter(result.values()), {})


def now_utc_timestamp() -> int:
    return int(time.time())


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

    headers = {"API-Key": api_key, "API-Sign": sig_digest.decode()}

    resp = requests.post(url, headers=headers, data=data, timeout=10)
    result = resp.json()
    if "error" in result and result["error"]:
        raise Exception(f"Kraken API error: {result['error']}")
    return result


def place_order(
    order_type: str,
    pair_code: str,
    fiat_symbol: str,
    amount: float,
    ledger_name: str,
    wallet_code: str,
    verbose: int = 0,
) -> dict:
    api_key, api_secret = load_kraken_keys()
    balance = _kraken_request("Balance", {}, api_key, api_secret).get("result", {})

    price_data = fetch_price_data(pair_code)
    price = float(price_data.get("c", [0])[0])
    coin_amount = round(amount / price, 8)
    usd_amount = coin_amount * price

    if order_type.lower() == "buy":
        available_usd = float(balance.get(fiat_symbol, 0.0))
        addlog(f"[DEBUG] Balance: {balance}", verbose_int=1, verbose_state=verbose)
        addlog(
            f"[DEBUG] Using fiat_symbol = {fiat_symbol}",
            verbose_int=1,
            verbose_state=verbose,
        )
        addlog(
            f"[DEBUG] available_usd = {available_usd} | trying to spend = {amount}",
            verbose_int=1,
            verbose_state=verbose,
        )
        if available_usd < amount:
            addlog(
                f"[SKIP] Insufficient {fiat_symbol}",
                verbose_int=1,
                verbose_state=verbose,
            )
            return {}
        addlog(
            f"[BUY ATTEMPT] {fiat_symbol} available: ${available_usd:.2f}, attempting to buy ${amount:.2f}",
            verbose_int=3,
            verbose_state=verbose,
        )
    else:
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
                "type": order_type,
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
                f"❗ Kraken rejected order due to insufficient funds on {ledger_name}.",
            )
            return {}
        raise

    txid_list = order_resp.get("result", {}).get("txid")
    txid = txid_list[0] if txid_list else None
    if not txid:
        raise Exception(f"{order_type.capitalize()} order failed — no txid returned")

    if order_type.lower() == "buy":
        post_balance = _kraken_request("Balance", {}, api_key, api_secret).get(
            "result", {}
        )
        new_fiat_balance = float(post_balance.get(fiat_symbol, 0.0))
        spent = float(balance.get(fiat_symbol, 0.0)) - new_fiat_balance
        if spent < amount * 0.8:
            for code, prev_amt in balance.items():
                if code == fiat_symbol:
                    continue
                drop = float(prev_amt) - float(post_balance.get(code, 0.0))
                if drop > amount * 0.8:
                    addlog(
                        f"[ERROR] Fiat mismatch: {code} decreased instead of {fiat_symbol}",
                        verbose_int=1,
                        verbose_state=verbose,
                    )
                    break
            else:
                addlog(
                    f"[ERROR] Fiat mismatch: {fiat_symbol} balance unchanged after buy",
                    verbose_int=1,
                    verbose_state=verbose,
                )

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
    pair_code: str,
    price: float,
    amount_usd: float,
    ledger_name: str,
    wallet_code: str,
    verbose: int = 0,
) -> dict:
    """Place a real buy order and normalise the result structure.

    Parameters are kept for API compatibility; ``client`` and ``price`` are
    currently unused as ``place_order`` pulls pricing from Kraken directly.
    """

    expected_raw = "USDC" if ledger_name.upper().endswith("USDC") else "USD"
    _, fiat_symbol = split_tag(pair_code)
    got_raw = fiat_symbol
    exp = norm_quote(expected_raw)
    got = norm_quote(got_raw)
    if exp != got:
        addlog(
            f"[DEBUG][QUOTE_RAW] expected_raw={expected_raw} got_raw={got_raw}",
            verbose_int=1,
            verbose_state=verbose,
        )
        addlog(
            f"[ABORT][QUOTE_MISMATCH] ledger={ledger_name} expected={exp} got={got} pair={pair_code}",
            verbose_int=1,
            verbose_state=verbose,
        )
        return {
            "error": "QUOTE_MISMATCH",
            "expected": exp,
            "got": got,
            "pair": pair_code,
        }

    result = place_order(
        "buy",
        pair_code,
        fiat_symbol,
        amount_usd,
        ledger_name,
        wallet_code,
        verbose,
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


def process_buy_signal(
    *,
    buy_signal: dict,
    ledger,
    t: int,
    runtime_state: dict,
    pair_code: str,
    price: float,
    ledger_name: str,
    wallet_code: str,
    verbose: int = 0,
):
    """Execute a buy and record it, setting the rebound gate on success."""

    result = execute_buy(
        None,
        pair_code=pair_code,
        price=price,
        amount_usd=buy_signal.get("size_usd", 0.0),
        ledger_name=ledger_name,
        wallet_code=wallet_code,
        verbose=verbose,
    )
    if not result or result.get("error"):
        return result

    note = apply_buy(
        ledger=ledger,
        window_name=buy_signal.get("window_name", ""),
        t=t,
        meta=buy_signal,
        result=result,
        state=runtime_state,
    )

    window_name = note.get("window_name")
    unlock_p = buy_signal.get("unlock_p")
    if window_name and unlock_p is not None:
        runtime_state.setdefault("buy_unlock_p", {})[window_name] = unlock_p

    msg = (
        f"[BUY][EXECUTED][{window_name} {note.get('window_size')}] "
        f"qty={note.get('entry_amount'):.6f} at ${note.get('entry_price'):.4f} "
        f"spend=${note.get('entry_usdt'):.2f} target=${note.get('target_price'):.4f} "
        f"p={note.get('p_buy'):.3f} note_id={note.get('id')}"
    )
    addlog(msg, verbose_int=1, verbose_state=verbose)
    send_telegram_message(msg)

    return result


def execute_sell(
    client,
    *,
    pair_code: str,
    coin_amount: float,
    price: float | None = None,
    ledger_name: str,
    verbose: int = 0,
) -> dict:
    """Place a real sell order and normalise the result structure.

    ``price`` is optional and, if absent, the current live price is fetched to
    estimate USD notional.
    """
    sell_price = price if price is not None else get_live_price(pair_code)
    usd_amount = coin_amount * sell_price
    _, fiat_symbol = split_tag(pair_code)
    expected_raw = "USDC" if ledger_name.upper().endswith("USDC") else "USD"
    got_raw = fiat_symbol
    exp = norm_quote(expected_raw)
    got = norm_quote(got_raw)
    if exp != got:
        addlog(
            f"[DEBUG][QUOTE_RAW] expected_raw={expected_raw} got_raw={got_raw}",
            verbose_int=1,
            verbose_state=verbose,
        )
        addlog(
            f"[ABORT][QUOTE_MISMATCH] ledger={ledger_name} expected={exp} got={got} pair={pair_code}",
            verbose_int=1,
            verbose_state=verbose,
        )
        return {
            "error": "QUOTE_MISMATCH",
            "expected": exp,
            "got": got,
            "pair": pair_code,
        }
    result = place_order(
        "sell",
        pair_code,
        fiat_symbol,
        usd_amount,
        ledger_name,
        "",
        verbose,
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
