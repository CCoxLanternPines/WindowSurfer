import base64
import hashlib
import hmac
import json
import requests
import time
from urllib.parse import urlencode

from systems.scripts.kraken_auth import load_kraken_keys
from systems.utils.addlog import addlog
from systems.utils.path import find_project_root

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
    """Return raw asset balances from Kraken."""

    api_key, api_secret = load_kraken_keys()
    result = _kraken_request("Balance", {}, api_key, api_secret).get("result", {})

    addlog(
        f"[INFO] Kraken balance fetched: {result}",
        verbose_int=3,
        verbose_state=verbose,
    )

    return {k: float(v) for k, v in result.items()}


def _load_snapshot(ledger_name: str) -> dict:
    root = find_project_root()
    snap_path = root / "data" / "snapshots" / f"{ledger_name}.json"
    if not snap_path.exists():
        raise FileNotFoundError(
            f"Snapshot for ledger '{ledger_name}' not found at {snap_path}"
        )
    with snap_path.open("r", encoding="utf-8") as f:
        snapshot = json.load(f)
    hour_start = int(time.time()) // 3600 * 3600
    if snapshot.get("timestamp", 0) < hour_start:
        raise FileNotFoundError(
            f"Snapshot for ledger '{ledger_name}' is stale"
        )
    return snapshot


def fetch_kraken_snapshot(ledger_name: str, verbose: int = 0) -> dict:
    api_key, api_secret = load_kraken_keys()
    try:
        balance_resp = _kraken_request("Balance", {}, api_key, api_secret).get(
            "result", {}
        )
        trades_resp = _kraken_request(
            "TradesHistory",
            {"type": "all", "trades": True},
            api_key,
            api_secret,
        ).get("result", {})

        snapshot = {
            "timestamp": int(time.time()),
            "balance": balance_resp,
            "trades": trades_resp.get("trades", trades_resp),
        }

        root = find_project_root()
        snap_dir = root / "data" / "snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)
        with (snap_dir / f"{ledger_name}.json").open("w", encoding="utf-8") as f:
            json.dump(snapshot, f)

        addlog(
            f"[SNAPSHOT] Cached Kraken balance and trades for {ledger_name}",
            verbose_int=3,
            verbose_state=verbose,
        )
        return snapshot
    except Exception as exc:
        addlog(
            f"[ERROR] Failed to fetch Kraken snapshot for {ledger_name}: {exc}",
            verbose_int=1,
            verbose_state=True,
        )
        return {}


def ensure_snapshot(ledger_name: str) -> dict:
    try:
        return _load_snapshot(ledger_name)
    except FileNotFoundError:
        addlog(f"[INFO] No snapshot found for {ledger_name}. Fetching...")
        return fetch_kraken_snapshot(ledger_name)
