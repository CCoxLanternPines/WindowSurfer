from __future__ import annotations

"""Utility for resolving symbol configuration across exchanges.

This module fetches asset metadata from Kraken and Binance and stores a
normalised cache in ``data/tmp/symbol_cache.json``. The cache is refreshed at
most once every 24 hours and provides a single source of truth for resolving
pair codes, wallet codes and exchange specific symbols.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

import requests

from systems.utils.path import find_project_root

CACHE_MAX_AGE = timedelta(hours=24)
KRAKEN_URL = "https://api.kraken.com/0/public/AssetPairs"
BINANCE_URL = "https://api.binance.com/api/v3/exchangeInfo"


def _cache_path() -> Path:
    root = find_project_root()
    path = root / "data" / "tmp" / "symbol_cache.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _is_cache_valid(path: Path) -> bool:
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return datetime.now() - mtime < CACHE_MAX_AGE


def _fetch_kraken_pairs() -> Dict[str, dict]:
    resp = requests.get(KRAKEN_URL, timeout=10)
    data = resp.json().get("result", {})
    pairs: Dict[str, dict] = {}
    for pair_code, info in data.items():
        altname = info.get("altname")
        wsname = info.get("wsname")
        base = info.get("base")
        quote = info.get("quote")
        if not altname or not wsname:
            continue
        base_asset, quote_asset = wsname.split("/")
        pairs[altname] = {
            "altname": altname,
            "pair_code": pair_code,
            "wsname": wsname,
            "wallet_code": base,
            "fiat": quote,
            "base_asset": base_asset,
            "quote_asset": quote_asset,
        }
    return pairs


def _fetch_binance_pairs() -> Dict[str, dict]:
    resp = requests.get(BINANCE_URL, timeout=10)
    data = resp.json().get("symbols", [])
    pairs: Dict[str, dict] = {}
    for info in data:
        symbol = info.get("symbol")
        base = info.get("baseAsset")
        quote = info.get("quoteAsset")
        if not symbol or not base or not quote:
            continue
        pairs[symbol] = {
            "altname": symbol,
            "symbol": symbol,
            "baseAsset": base,
            "quoteAsset": quote,
        }
    return pairs


def _build_cache() -> Dict[str, dict]:
    kraken = _fetch_kraken_pairs()
    binance = _fetch_binance_pairs()
    cache: Dict[str, dict] = {}
    for tag, kinfo in kraken.items():
        base = kinfo["base_asset"]
        quote = kinfo["quote_asset"]
        target_quote = "USDT" if quote == "USD" else quote
        binfo = binance.get(f"{base}{target_quote}")
        if not binfo:
            continue
        cache[tag] = {
            "kraken": {
                "altname": kinfo["altname"],
                "pair_code": kinfo["pair_code"],
                "wsname": kinfo["wsname"],
                "wallet_code": kinfo["wallet_code"],
                "fiat": kinfo["fiat"],
            },
            "binance": binfo,
        }
    return cache


_CACHE: Dict[str, dict] | None = None


def _load_cache() -> Dict[str, dict]:
    global _CACHE
    path = _cache_path()
    if _CACHE is not None and _is_cache_valid(path):
        return _CACHE
    if _is_cache_valid(path):
        with path.open("r", encoding="utf-8") as f:
            _CACHE = json.load(f)
            return _CACHE
    cache = _build_cache()
    with path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
    _CACHE = cache
    return cache


def get_symbol_config(tag: str) -> dict:
    """Return combined Kraken and Binance config for ``tag``.

    Parameters
    ----------
    tag:
        Symbol tag such as ``"SOLUSD"``.
    """
    tag = tag.upper()
    cache = _load_cache()
    cfg = cache.get(tag)
    if not cfg:
        raise ValueError(f"Symbol '{tag}' not found in cache")
    if "kraken" not in cfg or "binance" not in cfg:
        raise ValueError(f"Incomplete symbol data for '{tag}'")
    return cfg
