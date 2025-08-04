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

from systems.utils.loggers import logger
from systems.utils.path import find_project_root

CACHE_MAX_AGE = timedelta(hours=24)
KRAKEN_URL = "https://api.kraken.com/0/public/AssetPairs"
BINANCE_URL = "https://api.binance.com/api/v3/exchangeInfo"


alias_map = {
    "DOGE": "XDG",
    "XBT": "BTC",
    "USDT": "USDT",  # sanity
}

_reverse_alias_map = {v: k for k, v in alias_map.items()}


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


def _save_cache(cache: Dict[str, dict]) -> None:
    path = _cache_path()
    with path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def get_symbol_config(tag: str) -> dict:
    """Return combined Kraken and Binance config for ``tag``.

    Parameters
    ----------
    tag:
        Symbol tag such as ``"SOLUSD"``.
    """

    tag = tag.upper()
    cache = _load_cache()

    if tag in cache:
        return cache[tag]

    search_tag = tag
    for src, dst in alias_map.items():
        if search_tag.startswith(src):
            search_tag = dst + search_tag[len(src) :]
            break

    if search_tag in cache:
        cache[tag] = cache[search_tag]
        _save_cache(cache)
        logger.info(
            "[ALIAS MATCH] Resolved tag '%s' → Kraken altname '%s' → pair '%s'",
            tag,
            search_tag,
            cache[search_tag]["kraken"]["pair_code"],
        )
        return cache[tag]

    kraken_asset_pairs = requests.get(KRAKEN_URL, timeout=10).json().get(
        "result", {}
    )
    binance_pairs = _fetch_binance_pairs()

    for pair_key, data in kraken_asset_pairs.items():
        altname = data.get("altname", "").upper()
        wsname = data.get("wsname", "").replace("/", "").upper()
        base = data.get("base", "").replace("X", "").replace("Z", "").upper()
        quote = data.get("quote", "").replace("X", "").replace("Z", "").upper()
        concat = f"{base}{quote}"

        if search_tag in (altname, wsname, concat):
            base_ws, quote_ws = (data.get("wsname") or "").split("/")
            base_norm = _reverse_alias_map.get(base_ws.upper(), base_ws.upper())
            quote_norm = _reverse_alias_map.get(quote_ws.upper(), quote_ws.upper())
            target_quote = "USDT" if quote_norm == "USD" else quote_norm
            binfo = binance_pairs.get(f"{base_norm}{target_quote}")
            if not binfo:
                continue
            kcfg = {
                "altname": altname,
                "pair_code": pair_key,
                "wsname": data.get("wsname"),
                "wallet_code": data.get("base"),
                "fiat": data.get("quote"),
            }
            cfg = {"kraken": kcfg, "binance": binfo}
            cache[tag] = cfg
            _save_cache(cache)
            logger.info(
                "[ALIAS MATCH] Resolved tag '%s' → Kraken altname '%s' → pair '%s'",
                tag,
                altname,
                pair_key,
            )
            return cfg

    raise ValueError(f"Symbol '{tag}' not found in cache")


def ensure_all_symbols_loaded(settings: dict) -> None:
    """Validate that all ledger tags exist in the symbol cache.

    Parameters
    ----------
    settings:
        Settings dictionary containing ``ledger_settings``.

    Raises
    ------
    RuntimeError
        If any tags cannot be resolved after cache refresh.
    """

    missing: list[str] = []
    for ledger_cfg in settings.get("ledger_settings", {}).values():
        tag = ledger_cfg.get("tag")
        if not tag:
            continue
        try:
            get_symbol_config(tag)
        except ValueError:
            missing.append(tag)
    if missing:
        raise RuntimeError(
            f"[SYMBOL ERROR] The following tags are missing from Kraken/Binance: {missing}"
        )
