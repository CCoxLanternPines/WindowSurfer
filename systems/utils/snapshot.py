from __future__ import annotations

import json
import time
import hmac
import base64
import hashlib
from pathlib import Path
from urllib.parse import urlencode
from typing import Any, Dict

import requests

from .config import resolve_path


def load_snapshot(ledger_name: str) -> Dict[str, Any]:
    """Load a cached snapshot for ``ledger_name`` if it exists."""
    path = resolve_path(f"data/snapshots/{ledger_name}.json")
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


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
        "API-Sign": sig_digest.decode(),
    }

    resp = requests.post(url, headers=headers, data=data, timeout=10)
    result = resp.json()
    if "error" in result and result["error"]:
        raise Exception(f"Kraken API error: {result['error']}")
    return result


def fetch_snapshot_from_kraken(api_key: str, api_secret: str) -> Dict[str, Any]:
    """Fetch a fresh snapshot from Kraken."""
    balance_resp = _kraken_request("Balance", {}, api_key, api_secret).get(
        "result", {}
    )
    trades_resp = _kraken_request(
        "TradesHistory", {"type": "all", "trades": True}, api_key, api_secret
    ).get("result", {})
    return {
        "timestamp": int(time.time()),
        "balance": balance_resp,
        "trades": trades_resp.get("trades", trades_resp),
    }


def prime_snapshot(ledger_name: str, api_key: str, api_secret: str) -> Dict[str, Any]:
    """Fetch and cache a fresh snapshot for ``ledger_name``."""
    snap = fetch_snapshot_from_kraken(api_key, api_secret)
    path = resolve_path(f"data/snapshots/{ledger_name}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(snap, f, indent=2)
    return snap
