"""Kraken execution utilities.

This module provides thin wrappers around the Kraken REST API for placing
market orders.  Only the minimal functionality required by the live
engine is implemented and the functions are intentionally light-weight so
that they can be easily mocked during tests.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time
import urllib.parse
from typing import Any, Dict, Optional

import requests

from .wallet_cache import load_wallet_cache

_API_URL = "https://api.kraken.com"
_PRIVATE_PATH = "/0/private/AddOrder"


def _sign(secret: str, urlpath: str, data: Dict[str, Any]) -> str:
    postdata = urllib.parse.urlencode(data)
    encoded = (str(data["nonce"]).encode() + postdata.encode())
    message = urlpath.encode() + hashlib.sha256(encoded).digest()
    return base64.b64encode(hmac.new(base64.b64decode(secret), message, hashlib.sha512).digest()).decode()


def _request(api_key: str, api_secret: str, data: Dict[str, Any]) -> Dict[str, Any]:
    headers = {"API-Key": api_key, "API-Sign": _sign(api_secret, _PRIVATE_PATH, data)}
    resp = requests.post(_API_URL + _PRIVATE_PATH, data=data, headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _lookup_pair_info(pair: str) -> Dict[str, Any]:
    """Return precision and minimum size information for ``pair``."""
    caches = load_wallet_cache()["kraken"]
    for sym_map in caches.values():
        for info in sym_map.values():
            if info.get("pair") == pair:
                return info
    return {}


def _format_volume(pair: str, qty: float, *, precision: Optional[int], min_qty: Optional[float]) -> str:
    if min_qty is not None and qty < min_qty:
        qty = min_qty
    if precision is None:
        return str(qty)
    fmt = f"{{:.{precision}f}}"
    return fmt.format(qty)


def place_market_buy(pair: str, qty: float, *, precision: Optional[int] = None, min_qty: Optional[float] = None) -> Dict[str, Any]:
    """Place a market buy order on Kraken.

    Parameters are formatted according to precision and minimum quantity
    information sourced from the wallet cache.
    """

    api_key = os.getenv("KRAKEN_API_KEY", "")
    api_secret = os.getenv("KRAKEN_API_SECRET", "")
    if precision is None or min_qty is None:
        info = _lookup_pair_info(pair)
        precision = precision or info.get("quantity_precision")
        min_qty = min_qty or info.get("min_order_coin")
    data = {
        "nonce": int(time.time() * 1000),
        "pair": pair,
        "type": "buy",
        "ordertype": "market",
        "volume": _format_volume(pair, qty, precision=precision, min_qty=min_qty),
    }
    if not api_key or not api_secret:
        return data  # pragma: no cover - used in dry-run/testing
    return _request(api_key, api_secret, data)


def place_market_sell(pair: str, qty: float, *, precision: Optional[int] = None, min_qty: Optional[float] = None) -> Dict[str, Any]:
    """Place a market sell order on Kraken."""

    api_key = os.getenv("KRAKEN_API_KEY", "")
    api_secret = os.getenv("KRAKEN_API_SECRET", "")
    if precision is None or min_qty is None:
        info = _lookup_pair_info(pair)
        precision = precision or info.get("quantity_precision")
        min_qty = min_qty or info.get("min_order_coin")
    data = {
        "nonce": int(time.time() * 1000),
        "pair": pair,
        "type": "sell",
        "ordertype": "market",
        "volume": _format_volume(pair, qty, precision=precision, min_qty=min_qty),
    }
    if not api_key or not api_secret:
        return data  # pragma: no cover
    return _request(api_key, api_secret, data)


__all__ = ["place_market_buy", "place_market_sell"]
