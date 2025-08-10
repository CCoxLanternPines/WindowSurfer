import os
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DATA_RAW = Path("data/raw")

_UNITS = {"d": 86400, "w": 604800, "m": 2592000, "y": 31536000}


def _parse_timespan(expr: str) -> int:
    """Return span in seconds from shorthand like '1d', '2w', '3m', '4y'."""
    if expr is None:
        return 0
    m = re.fullmatch(r"(\d+)([dwmy])", expr.strip())
    if not m:
        raise ValueError(f"invalid timespan: {expr}")
    value, unit = m.groups()
    return int(value) * _UNITS[unit]


def load_candle_history(symbol: str, start: str | None = None, range: str | None = None) -> pd.DataFrame:
    """Load historical candles from Parquet and apply optional windowing."""
    path = DATA_RAW / f"{symbol.upper()}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"history not found for {symbol}")

    df = pd.read_parquet(path)
    if df.empty:
        return df

    df.sort_values("timestamp", inplace=True)
    now = int(datetime.now(timezone.utc).timestamp())

    start_ts = df["timestamp"].min()
    if start:
        start_ts = now - _parse_timespan(start)

    end_ts = df["timestamp"].max()
    if range:
        end_ts = start_ts + _parse_timespan(range)

    window = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]
    return window.reset_index(drop=True)


def merge_live_updates(df: pd.DataFrame, live_df: pd.DataFrame) -> pd.DataFrame:
    """Append live candles to history, returning a sorted de-duplicated frame."""
    combined = pd.concat([df, live_df], ignore_index=True)
    combined.drop_duplicates(subset="timestamp", keep="last", inplace=True)
    combined.sort_values("timestamp", inplace=True)
    return combined.reset_index(drop=True)
