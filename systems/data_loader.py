from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import ccxt
import pandas as pd

from .paths import RAW_DIR, raw_parquet, ensure_dirs


logger = logging.getLogger(__name__)

CACHE_DIR = RAW_DIR


def _load_settings() -> dict:
    path = Path("settings.json")
    if path.exists():
        with path.open() as fh:
            return json.load(fh)
    return {}


def _resolve_symbol(tag: str, exchange: str) -> str:
    settings = _load_settings()
    return settings.get(tag, {}).get(f"{exchange}_name", tag)


def clean_candles(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize candle DataFrame.

    Ensures ascending order, unique timestamps and exactly 1h spacing.
    Missing candles are forward filled with NaNs.
    """
    if df.empty:
        return df

    # sort and deduplicate
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last")

    # convert timestamp to UTC datetime
    unit = "ms" if df["timestamp"].max() > 10**10 else "s"
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit=unit, utc=True)

    # create continuous 1h range and reindex
    full_range = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="1h")
    df = df.set_index("timestamp").reindex(full_range)
    df.index.name = "timestamp"
    df = df.reset_index()

    # verify spacing
    diffs = df["timestamp"].diff().dropna()
    if not (diffs == pd.Timedelta(hours=1)).all():
        raise ValueError("Non 1h candle intervals detected after cleaning")

    # convert back to seconds
    df["timestamp"] = (df["timestamp"].astype("int64") // 10**9).astype(int)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def fetch_all_history_binance(symbol: str) -> pd.DataFrame:
    """Fetch complete 1h history for ``symbol`` from Binance."""
    exchange = ccxt.binance()
    timeframe = "1h"
    limit = 1000
    since = 0
    all_ohlcv = []

    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not ohlcv:
            break
        all_ohlcv.extend(ohlcv)
        since = ohlcv[-1][0] + 3600 * 1000
        if len(ohlcv) < limit:
            break

    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    return clean_candles(df)


def fetch_range_kraken(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch ``symbol`` candles for the range ``[start, end]`` from Kraken."""
    exchange = ccxt.kraken()
    timeframe = "1h"
    limit = 720

    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)

    since = start_ms
    all_ohlcv = []
    while since < end_ms:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not ohlcv:
            break
        all_ohlcv.extend(ohlcv)
        last_ts = ohlcv[-1][0]
        since = last_ts + 3600 * 1000
        if last_ts >= end_ms:
            break

    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df[df["timestamp"] <= end_ms]
    return clean_candles(df)


def load_or_fetch(tag: str, fetch_all: bool = False, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """Load candles for ``tag`` from cache or fetch them if needed."""
    ensure_dirs()
    cache_path = raw_parquet(tag)
    binance_symbol = _resolve_symbol(tag, "binance")
    kraken_symbol = _resolve_symbol(tag, "kraken")

    if fetch_all:
        df = fetch_all_history_binance(binance_symbol)
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        return df

    if start and end:
        return fetch_range_kraken(kraken_symbol, start, end)
    if start or end:
        raise ValueError("Both start and end must be provided")

    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        return clean_candles(df)

    df = fetch_all_history_binance(binance_symbol)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df
