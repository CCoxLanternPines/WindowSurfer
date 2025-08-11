from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import ccxt

from systems.utils.config import resolve_path
from systems.utils.addlog import addlog

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

    ts = combined["timestamp"].to_numpy()
    if len(ts) > 1:
        gaps = (ts[1:] - ts[:-1]) != 3600
        if gaps.any():
            missing_spans = int(gaps.sum())
            addlog(
                f"[WARN] Post-merge gaps detected: {missing_spans} hour(s) missing",
                verbose_state=True,
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    return len(combined)


def get_raw_path_for_coin(coin: str, ext: str = "csv") -> Path:
    """Return the full path to the raw-data file for a given coin."""
    root = resolve_path("")
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir / f"{coin.strip().upper()}.{ext}"


def compute_missing_ranges(
    candles_df: pd.DataFrame, start_ts: int, end_ts: int, interval_ms: int
) -> List[tuple[int, int]]:
    """Return a list of (start, end) tuples for missing candle ranges."""
    interval_s = interval_ms // 1000
    if candles_df.empty:
        return [(start_ts, end_ts)]

    windowed = candles_df[
        (candles_df["timestamp"] >= start_ts)
        & (candles_df["timestamp"] <= end_ts)
    ]["timestamp"].sort_values()

    ranges: List[tuple[int, int]] = []
    current = start_ts
    for ts in windowed:
        if ts > current:
            ranges.append((current, min(ts - interval_s, end_ts)))
        current = ts + interval_s
        if current > end_ts:
            break

    if current <= end_ts:
        ranges.append((current, end_ts))

    return ranges


def fetch_range(
    exchange_name: str, tag: str, start_ts: int, end_ts: int
) -> pd.DataFrame:
    """Fetch candles for ``tag`` on ``exchange_name`` within [start_ts, end_ts]."""
    start_ms = int(start_ts * 1000)
    end_ms = int(end_ts * 1000)

    if exchange_name.lower() == "kraken":
        rows = _fetch_kraken(tag, start_ms, end_ms)
    elif exchange_name.lower() == "binance":
        rows = _fetch_binance(tag, start_ms, end_ms)
    else:
        raise ValueError(f"Unknown exchange '{exchange_name}'")

    df = pd.DataFrame(rows, columns=COLUMNS)
    if not df.empty:
        df["timestamp"] = (
            pd.to_datetime(df["timestamp"], unit="ms", utc=True).astype("int64")
            // 1_000_000_000
        )
        df["timestamp"] = (df["timestamp"] // 3600) * 3600
        df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]
        df[COLUMNS] = df[COLUMNS].apply(pd.to_numeric, errors="coerce")
        df = df.dropna(subset=["timestamp"])
    return df
