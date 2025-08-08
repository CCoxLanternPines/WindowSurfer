import requests

from systems.scripts.kraken_auth import load_kraken_keys
from systems.utils.addlog import addlog
from systems.utils.price_fetcher import get_price
from systems.utils.snapshot import load_snapshot, prime_snapshot

KRAKEN_API_URL = "https://api.kraken.com"


def get_live_price(kraken_pair: str) -> float:
    """Return the current best ask price for ``kraken_pair``."""
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


def get_kraken_balance(fiat_code: str, verbose: int = 0) -> dict:
    """Return Kraken balances converted to ``fiat_code`` for logging."""
    from systems.scripts.execution_handler import _kraken_request

    api_key, api_secret = load_kraken_keys()
    result = _kraken_request("Balance", {}, api_key, api_secret).get("result", {})

    fiat_code = fiat_code.upper()
    fiat_code_symbol = fiat_code[1:] if fiat_code.startswith("Z") else fiat_code

    fiat_balances = {}
    for asset, amount in result.items():
        amount = float(amount)
        if asset.upper() == fiat_code:
            fiat_balances[asset] = amount
        else:
            price = get_price(f"{asset}{fiat_code_symbol}")
            if price:
                fiat_balances[asset] = amount * price

    addlog(
        f"[INFO] Kraken balance fetched ({fiat_code_symbol}): {fiat_balances}",
        verbose_int=3,
        verbose_state=verbose,
    )

    return {k: float(v) for k, v in result.items()}


def ensure_snapshot(ledger_name: str) -> dict:
    """Load a snapshot or fetch a new one if absent."""
    snapshot = load_snapshot(ledger_name)
    if snapshot:
        return snapshot
    api_key, api_secret = load_kraken_keys()
    return prime_snapshot(ledger_name, api_key, api_secret)
