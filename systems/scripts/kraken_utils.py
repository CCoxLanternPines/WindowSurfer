import os
import requests
from pathlib import Path
import yaml

from systems.utils.addlog import addlog
from systems.utils.price_fetcher import get_price
from systems.utils.snapshot import load_snapshot, prime_snapshot

KRAKEN_API_URL = "https://api.kraken.com"


# --- Authentication -------------------------------------------------------

def load_kraken_keys(path: str = "kraken.yaml") -> tuple[str, str]:
    """Load Kraken API key/secret using WS_ACCOUNT or fallback YAML."""
    account = os.environ.get("WS_ACCOUNT")
    if account:
        try:
            from systems.utils.load_config import load_config

            cfg = load_config()
            acct = cfg.get("accounts", {}).get(account, {})
            key = acct.get("api_key")
            secret = acct.get("api_secret")
            if key and secret:
                return key, secret
        except Exception:
            pass

    file = Path(path)
    if not file.exists():
        raise FileNotFoundError("Missing kraken.yaml in project root")

    with file.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    kraken = data.get("kraken")
    if not kraken or "api_key" not in kraken or "api_secret" not in kraken:
        raise ValueError("Malformed kraken.yaml: missing 'kraken.api_key' or 'api_secret'")

    return kraken["api_key"], kraken["api_secret"]


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

