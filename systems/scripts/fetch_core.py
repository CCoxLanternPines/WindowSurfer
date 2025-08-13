from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pandas as pd
import ccxt

from systems.utils.config import resolve_path
from systems.utils.addlog import addlog

COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def get_coin_raw_path(coin: str, ext: str = "csv") -> Path:
    """Return the raw-data path for a given ``coin``."""
    root = resolve_path("")
    return root / "data" / "raw" / f"{coin.upper()}.{ext}"


def _fetch_kraken(symbol: str, start_ms: int, end_ms: int) -> List[List]:
    exchange = ccxt.kraken({"enableRateLimit": True})
    limit = min(720, int((end_ms - start_ms) // 3600000) + 1)
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe="1h", since=start_ms, limit=limit)
    return [row for row in ohlcv if row and start_ms <= row[0] <= end_ms]



def _iso(ms: int) -> str:
    try:
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S%z"
        )
    except Exception:
        return str(ms)


def _vprint(enabled: bool, msg: str):
    if enabled:
        print(msg)


def _fetch_binance(symbol: str, start_ms: int, end_ms: int) -> List[List]:
    """
    Forward pagination (old -> new) with page-by-page debug and simple retries.
    Returns a list of [ts, open, high, low, close, volume] rows.
    """
    exchange = ccxt.binance({"enableRateLimit": True})
    rows: List[List] = []
    limit = 1000
    page_idx = 0
    verbose = os.environ.get("FETCH_DEBUG", "0") == "1"
    MAX_EMPTY_RETRY = 3
    PARTIAL_RETRY_ON_LEN_LT = limit  # retry once if page smaller than limit
    SLEEP_ON_RETRY_SEC = 0.5

    since = start_ms
    last_progress_ts: int | None = None

    while since <= end_ms:
        page = (
            exchange.fetch_ohlcv(symbol, timeframe="1h", since=since, limit=limit)
            or []
        )
        page_len = len(page)

        empty_retries = 0
        while page_len == 0 and empty_retries < MAX_EMPTY_RETRY:
            _vprint(
                verbose,
                f"[FETCH][BINANCE] page={page_idx} since={since} EMPTY (retry {empty_retries+1}/{MAX_EMPTY_RETRY})",
            )
            time.sleep(SLEEP_ON_RETRY_SEC)
            page = (
                exchange.fetch_ohlcv(symbol, timeframe="1h", since=since, limit=limit)
                or []
            )
            page_len = len(page)
            empty_retries += 1

        if page_len == 0:
            _vprint(
                verbose, f"[FETCH][BINANCE] floor or no data at since={since} → stop"
            )
            break

        did_partial_retry = False
        if page_len < PARTIAL_RETRY_ON_LEN_LT:
            _vprint(
                verbose,
                f"[FETCH][BINANCE] page={page_idx} got {page_len}/{limit} rows → partial, retrying once",
            )
            time.sleep(SLEEP_ON_RETRY_SEC)
            page2 = (
                exchange.fetch_ohlcv(symbol, timeframe="1h", since=since, limit=limit)
                or []
            )
            if len(page2) > page_len:
                page = page2
                page_len = len(page)
                did_partial_retry = True

        first_ts = page[0][0]
        last_ts = page[-1][0]
        filtered = [r for r in page if start_ms <= r[0] <= end_ms]
        rows.extend(filtered)
        page_idx += 1

        _vprint(
            verbose,
            f"[FETCH][BINANCE] page={page_idx:05d} since={since} "
            f"first={_iso(first_ts)} last={_iso(last_ts)} rows={page_len} total={len(rows)}"
            f"{' (partial→retry improved)' if did_partial_retry else ''}",
        )

        next_since = last_ts + 1
        if last_progress_ts is not None and next_since <= last_progress_ts:
            _vprint(
                verbose,
                f"[FETCH][BINANCE] no progress (next_since={next_since} <= last_progress={last_progress_ts}) → stop",
            )
            break
        last_progress_ts = next_since
        since = next_since

    return rows


def _to_df(rows: List[List], start_ts: int, end_ts: int) -> pd.DataFrame:
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


def fetch_kraken_range(symbol: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    rows = _fetch_kraken(symbol, int(start_ts * 1000), int(end_ts * 1000))
    return _to_df(rows, start_ts, end_ts)


def fetch_binance_range(symbol: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    rows = _fetch_binance(symbol, int(start_ts * 1000), int(end_ts * 1000))
    return _to_df(rows, start_ts, end_ts)


def fetch_last_720_kraken(symbol: str, end_ts: int) -> pd.DataFrame:
    """Return the last 720 hourly candles from Kraken for ``symbol``."""
    start_ts = end_ts - 719 * 3600
    return fetch_kraken_range(symbol, start_ts, end_ts)


def heal_recent(existing: pd.DataFrame, recent: pd.DataFrame) -> tuple[pd.DataFrame, int, int, int]:
    """Merge ``recent`` candles into ``existing`` and report stats.

    Returns
    -------
    tuple[pd.DataFrame, int, int, int]
        ``(merged_df, appended, deduped, gaps)`` where ``gaps`` is the
        number of missing hourly candles in the latest 720-window.
    """

    combined = pd.concat([existing, recent], ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset="timestamp").sort_values("timestamp")
    appended = len(combined) - len(existing)
    dedup = before - len(combined)

    gaps = 0
    if not recent.empty:
        end_ts = int(recent["timestamp"].max())
        start_ts = end_ts - 719 * 3600
        window = combined[combined["timestamp"] >= start_ts]
        expected = set(range(start_ts, end_ts + 3600, 3600))
        gaps = len(expected - set(window["timestamp"].astype(int)))

    return combined, appended, dedup, gaps


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


def get_raw_path(tag: str, ext: str = "csv") -> Path:
    """Return the full path to the raw-data file for a given tag."""
    root = resolve_path("")
    return root / "data" / "raw" / f"{tag}.{ext}"


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
    if exchange_name.lower() == "kraken":
        return fetch_kraken_range(tag, start_ts, end_ts)
    if exchange_name.lower() == "binance":
        return fetch_binance_range(tag, start_ts, end_ts)
    raise ValueError(f"Unknown exchange '{exchange_name}'")
