from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time
import urllib.parse
from typing import Any, Dict

import requests


def _get_api_keys() -> tuple[str, str]:
    """Return Kraken API key and secret from environment variables."""
    api_key = os.getenv("KRAKEN_API_KEY", "")
    api_secret = os.getenv("KRAKEN_API_SECRET", "")
    if not api_key or not api_secret:
        raise RuntimeError("Kraken API credentials not found in environment")
    return api_key, api_secret


def _kraken_request(endpoint: str, data: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Send a signed request to the Kraken private API."""
    api_key, api_secret = _get_api_keys()

    url_path = f"/0/private/{endpoint}"
    url = f"https://api.kraken.com{url_path}"
    nonce = str(int(time.time() * 1000))
    post_data = data.copy() if data else {}
    post_data["nonce"] = nonce
    encoded = urllib.parse.urlencode(post_data)
    message = (nonce + encoded).encode()
    sha = hashlib.sha256(message).digest()
    mac = hmac.new(base64.b64decode(api_secret), url_path.encode() + sha, hashlib.sha512)
    sig = base64.b64encode(mac.digest())

    headers = {"API-Key": api_key, "API-Sign": sig.decode()}
    resp = requests.post(url, headers=headers, data=post_data, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_kraken_balance() -> Dict[str, float]:
    """Fetch account balances from Kraken."""
    resp = _kraken_request("Balance")
    result = resp.get("result", {})
    # Convert all balances to float
    return {k: float(v) for k, v in result.items()}


def buy_order(pair: str, volume: float) -> Dict[str, Any]:
    """Place a market buy order and return detailed fill information."""
    order = _kraken_request(
        "AddOrder",
        {"pair": pair, "type": "buy", "ordertype": "market", "volume": volume, "trades": True},
    )
    txid_list = order.get("result", {}).get("txid", [])
    txid = txid_list[0] if txid_list else None
    if not txid:
        raise RuntimeError("Failed to retrieve txid from buy order response")
    details = _kraken_request("QueryOrders", {"txid": txid, "trades": True})
    info = details.get("result", {}).get(txid, {})
    return {
        "price": float(info.get("price", 0.0)),
        "amount": float(info.get("vol_exec", 0.0)),
        "cost": float(info.get("cost", 0.0)),
        "fee": float(info.get("fee", 0.0)),
        "ts": float(info.get("closetm", info.get("opentm", 0.0))),
        "txid": txid,
    }


def sell_order(pair: str, volume: float) -> Dict[str, Any]:
    """Place a market sell order and return detailed fill information."""
    order = _kraken_request(
        "AddOrder",
        {"pair": pair, "type": "sell", "ordertype": "market", "volume": volume, "trades": True},
    )
    txid_list = order.get("result", {}).get("txid", [])
    txid = txid_list[0] if txid_list else None
    if not txid:
        raise RuntimeError("Failed to retrieve txid from sell order response")
    details = _kraken_request("QueryOrders", {"txid": txid, "trades": True})
    info = details.get("result", {}).get(txid, {})
    return {
        "price": float(info.get("price", 0.0)),
        "amount": float(info.get("vol_exec", 0.0)),
        "cost": float(info.get("cost", 0.0)),
        "fee": float(info.get("fee", 0.0)),
        "ts": float(info.get("closetm", info.get("opentm", 0.0))),
        "txid": txid,
    }
