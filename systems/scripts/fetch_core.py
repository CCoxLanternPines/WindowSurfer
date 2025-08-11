from __future__ import annotations

"""Core helpers for deterministic, gapless historical fetches."""

from pathlib import Path
from typing import Iterable, List, Tuple

import ccxt
import pandas as pd

from systems.utils.config import resolve_path

# Canonical column order for 1h OHLCV dataframes.
COLUMNS = ["ts", "open", "high", "low", "close", "volume"]


class FetchAbort(RuntimeError):
    """Raised when a deterministic fetch cannot be completed."""


# ---------------------------------------------------------------------------
# fetching
# ---------------------------------------------------------------------------

def _fetch_kraken(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Return raw Kraken candles in the requested span.

    The caller is responsible for canonicalising the dataframe. When no data
    is available an empty dataframe is returned instead of raising.
    """

    exchange = ccxt.kraken({"enableRateLimit": True})
    limit = min(720, int((end_ms - start_ms) // 3_600_000))
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe="1h", since=start_ms, limit=limit)
    except Exception:
        ohlcv = []
    rows = [row for row in ohlcv if row and start_ms <= row[0] < end_ms]
    return pd.DataFrame(rows, columns=COLUMNS)


def _fetch_binance(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Return raw Binance candles in the requested span.

    Fetches data in 1000 row chunks. An empty dataframe is returned when no
    data is available; the caller must handle such cases gracefully.
    """

    exchange = ccxt.binance({"enableRateLimit": True})
    limit = 1_000
    rows: List[List] = []
    current = start_ms
    while current < end_ms:
        try:
            chunk = exchange.fetch_ohlcv(symbol, timeframe="1h", since=current, limit=limit)
        except Exception:
            chunk = []
        if not chunk:
            break
        filtered = [r for r in chunk if r and current <= r[0] < end_ms]
        if not filtered:
            break
        rows.extend(filtered)
        last = filtered[-1][0]
        current = last + 3_600_000
    return pd.DataFrame(rows, columns=COLUMNS)


# ---------------------------------------------------------------------------
# dataframe helpers
# ---------------------------------------------------------------------------

def canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure canonical columns and hourly UTC timestamps in ms."""

    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    df = df.copy()
    df.columns = COLUMNS
    df["ts"] = (df["ts"].astype("int64") // 3_600_000) * 3_600_000
    df = df.drop_duplicates(subset="ts").sort_values("ts")
    return df.reset_index(drop=True)


def _find_gaps(ts: Iterable[int]) -> List[Tuple[int, int, int]]:
    ts_list = sorted(ts)
    gaps: List[Tuple[int, int, int]] = []
    for prev, nxt in zip(ts_list, ts_list[1:]):
        if nxt - prev > 3_600_000:
            start = prev + 3_600_000
            end = nxt - 3_600_000
            k = int((end - start) // 3_600_000 + 1)
            gaps.append((start, end, k))
    return gaps


def reindex_hourly(
    df: pd.DataFrame, start_ms: int, end_ms: int
) -> Tuple[pd.DataFrame, List[Tuple[int, int, int]]]:
    """Reindex ``df`` to an hourly grid and return gaps."""

    idx = range(start_ms, end_ms, 3_600_000)
    if df.empty:
        full = pd.DataFrame(index=idx, columns=COLUMNS[1:])
        full.index.name = "ts"
        full = full.reset_index()
        gap_end = end_ms - 3_600_000 if end_ms > start_ms else start_ms
        gaps = [(start_ms, gap_end, len(idx))] if len(idx) else []
        return full, gaps

    df = df.set_index("ts").reindex(idx)
    df.index.name = "ts"
    full = df.reset_index()
    gaps = []
    i = 0
    while i < len(full):
        if pd.isna(full.loc[i, "open"]):
            gap_start = full.loc[i, "ts"]
            while i < len(full) and pd.isna(full.loc[i, "open"]):
                i += 1
            gap_end = full.loc[i - 1, "ts"]
            k = int((gap_end - gap_start) // 3_600_000 + 1)
            gaps.append((gap_start, gap_end, k))
        else:
            i += 1
    return full, gaps


# ---------------------------------------------------------------------------
# path helper
# ---------------------------------------------------------------------------

def get_raw_path(tag: str) -> Path:
    """Return the canonical raw-data path for ``tag``."""

    root = resolve_path("")
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir / f"{tag}_1h.parquet"


