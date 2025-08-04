#!/usr/bin/env python3
"""Fetch and cache symbol metadata for Kraken and Binance."""

from __future__ import annotations

import json
from pathlib import Path

import requests

SNAPSHOT_DIR = Path("data/snapshots")
KRAKEN_URL = "https://api.kraken.com/0/public/AssetPairs"
BINANCE_URL = "https://api.binance.com/api/v3/exchangeInfo"
PROXIES = {"http": "", "https": ""}


def _sync_kraken() -> dict:
    resp = requests.get(KRAKEN_URL, timeout=30, proxies=PROXIES)
    resp.raise_for_status()
    pairs = resp.json().get("result", {})
    out: dict[str, dict] = {}
    for pair in pairs.values():
        altname = pair.get("altname")
        if not altname:
            continue
        out[altname] = {
            "kraken_tag": altname,
            "kraken_pair": pair.get("pair"),
            "kraken_name": pair.get("wsname"),
            "wallet_code": pair.get("base"),
            "fiat_code": pair.get("quote"),
        }
    return out


def _sync_binance() -> dict:
    resp = requests.get(BINANCE_URL, timeout=30, proxies=PROXIES)
    resp.raise_for_status()
    symbols = resp.json().get("symbols", [])
    out: dict[str, dict] = {}
    for sym in symbols:
        base = sym.get("baseAsset")
        quote = sym.get("quoteAsset")
        if not base or not quote:
            continue
        canonical_quote = "USD" if quote == "USDT" else quote
        tag = f"{base}{canonical_quote}"
        out[tag] = {
            "binance_tag": sym.get("symbol"),
            "base_asset": base,
            "quote_asset": quote,
        }
    return out


def main() -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    kraken_symbols = _sync_kraken()
    (SNAPSHOT_DIR / "kraken_symbols.json").write_text(
        json.dumps(kraken_symbols, indent=2, sort_keys=True)
    )

    binance_symbols = _sync_binance()
    (SNAPSHOT_DIR / "binance_symbols.json").write_text(
        json.dumps(binance_symbols, indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()
