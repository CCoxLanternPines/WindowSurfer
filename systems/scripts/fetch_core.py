from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import ccxt

from systems.utils.path import find_project_root

COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def _fetch_kraken(symbol: str, start_ms: int, end_ms: int) -> List[List]:
    exchange = ccxt.kraken({"enableRateLimit": True})
    limit = min(720, int((end_ms - start_ms) // 3600000) + 1)
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe="1h", since=start_ms, limit=limit)
    return [row for row in ohlcv if row and start_ms <= row[0] <= end_ms]


def _fetch_binance(symbol: str, start_ms: int, end_ms: int) -> List[List]:
    exchange = ccxt.binance({"enableRateLimit": True})
    limit = 1000
    rows: List[List] = []
    current_end = end_ms
    while current_end >= start_ms:
        since = max(start_ms, current_end - limit * 3600000)
        chunk = exchange.fetch_ohlcv(symbol, timeframe="1h", since=since, limit=limit)
        if not chunk:
            break
        filtered = [r for r in chunk if r and since <= r[0] <= current_end]
        rows.extend(filtered)
        earliest = filtered[0][0]
        if earliest <= start_ms:
            break
        current_end = earliest - 3600000
    return rows


def _load_existing(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        return df
    return pd.DataFrame(columns=COLUMNS)


def _merge_and_save(path: Path, existing: pd.DataFrame, new_frames: List[pd.DataFrame]) -> int:
    combined = pd.concat([existing] + new_frames, ignore_index=True)
    combined = combined.drop_duplicates(subset="timestamp").sort_values("timestamp")
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    return len(combined)


def get_raw_path(tag: str, ext: str = "csv") -> Path:
    """Return the full path to the raw-data file for a given tag."""
    root = find_project_root()
    return root / "data" / "raw" / f"{tag}.{ext}"
