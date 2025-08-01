import requests
import time
import hashlib
import hmac
import base64
from urllib.parse import urlencode

from systems.scripts.kraken_auth import load_kraken_keys
from systems.utils.logger import addlog
from systems.utils.price_fetcher import get_price

KRAKEN_API_URL = "https://api.kraken.com"


def get_live_price(kraken_pair: str) -> float:
    """Return the current best ask price for ``kraken_pair``.

    Parameters
    ----------
    kraken_pair:
        Asset pair in Kraken's format (e.g. ``"DOGE/USD"`` or ``"DOGEUSD"``).

    Returns
    -------
    float
        The best ask price if available, otherwise ``0.0``.
    """
    pair = kraken_pair.replace("/", "")
    try:
        resp = requests.get(
            f"{KRAKEN_API_URL}/0/public/Ticker", params={"pair": pair}, timeout=10
        )
        data = resp.json()
        result = next(iter(data.get("result", {}).values()), {})
        ask = result.get("a", [None])[0]
        return float(ask) if ask is not None else 0.0
    except Exception:
        return 0.0

def _kraken_request(endpoint: str, data: dict, api_key: str, api_secret: str) -> dict:
    url_path = f"/0/private/{endpoint}"
    url = KRAKEN_API_URL + url_path

    nonce = str(int(1000 * time.time()))
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

    resp = requests.post(url, headers=headers, data=data)
    result = resp.json()

    if "error" in result and result["error"]:
        raise Exception(f"Kraken API error: {result['error']}")

    return result


def get_kraken_balance(verbose: int = 0) -> dict:
    # Import here to avoid circular dependency when execution_handler imports this module
    from systems.scripts.execution_handler import _kraken_request

    api_key, api_secret = load_kraken_keys()
    result = _kraken_request("Balance", {}, api_key, api_secret).get("result", {})

    usd_balances = {}
    for asset, amount in result.items():
        amount = float(amount)
        if asset.upper() in {"ZUSD", "USD", "USDT"}:
            usd_balances[asset] = amount
        else:
            price = get_price(f"{asset}USD")
            if price:
                usd_balances[asset] = amount * price

    addlog(
        f"[INFO] Kraken balance fetched (USD): {usd_balances}",
        verbose_int=3,
        verbose_state=verbose,
    )

    return {k: float(v) for k, v in result.items()}
