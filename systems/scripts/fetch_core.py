import json
import os
import time
from pathlib import Path
from typing import Dict

import ccxt
import pandas as pd

DATA_RAW = Path("data/raw")
DATA_META = Path("data/meta")


def _load_market_cache(name: str) -> Dict[str, dict]:
    """Load cached market info for an exchange."""
    path = DATA_META / f"{name}_pairs.json"
    if not path.exists():
        raise FileNotFoundError(f"wallet cache missing: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _resolve_pair(cache: Dict[str, dict], symbol: str, fiat: str) -> str:
    """Return exchange pair for symbol/fiat from cached markets."""
    symbol = symbol.upper()
    fiat = fiat.upper()
    for pair in cache.keys():
        try:
            base, quote = pair.split("/")
        except ValueError:
            continue
        if base == symbol and quote == fiat:
            return pair
    raise ValueError(f"pair for {symbol}/{fiat} not found in wallet cache")


def fetch_full_history(symbol: str, fiat: str) -> None:
    """Fetch full 1h candle history from Binance and store to Parquet."""
    markets = _load_market_cache("binance")
    pair = _resolve_pair(markets, symbol, fiat)
    exchange = ccxt.binance({"enableRateLimit": True})

    since = None
    all_candles = []
    while True:
        candles = exchange.fetch_ohlcv(pair, timeframe="1h", since=since, limit=1000)
        if not candles:
            break
        all_candles.extend(candles)
        since = candles[-1][0] + 3600_000
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = df["timestamp"] // 1000
    df.sort_values("timestamp", inplace=True)

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA_RAW / f"{symbol.upper()}.parquet", index=False)


def fetch_update_history(symbol: str, fiat: str) -> None:
    """Append recent 1h candles from Kraken to existing history."""
    path = DATA_RAW / f"{symbol.upper()}.parquet"
    if not path.exists():
        fetch_full_history(symbol, fiat)
        return

    existing = pd.read_parquet(path)
    last_ts = int(existing["timestamp"].max()) if not existing.empty else 0

    markets = _load_market_cache("kraken")
    pair = _resolve_pair(markets, symbol, fiat)
    exchange = ccxt.kraken({"enableRateLimit": True})
    candles = exchange.fetch_ohlcv(pair, timeframe="1h", limit=120)
    update = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    update["timestamp"] = update["timestamp"] // 1000

    if last_ts and update["timestamp"].min() - last_ts > 120 * 3600:
        fetch_full_history(symbol, fiat)
        return

    combined = pd.concat([existing, update], ignore_index=True)
    combined.drop_duplicates(subset="timestamp", keep="last", inplace=True)
    combined.sort_values("timestamp", inplace=True)
    combined.to_parquet(path, index=False)


def build_wallet_cache() -> None:
    """Fetch market metadata from Binance and Kraken and cache locally."""
    DATA_META.mkdir(parents=True, exist_ok=True)

    binance = ccxt.binance({"enableRateLimit": True})
    kraken = ccxt.kraken({"enableRateLimit": True})

    binance_markets = binance.load_markets()
    kraken_markets = kraken.load_markets()

    with open(DATA_META / "binance_pairs.json", "w", encoding="utf-8") as fh:
        json.dump(binance_markets, fh, indent=2)

    with open(DATA_META / "kraken_pairs.json", "w", encoding="utf-8") as fh:
        json.dump(kraken_markets, fh, indent=2)
