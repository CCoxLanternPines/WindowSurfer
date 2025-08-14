from __future__ import annotations

"""Unified candle fetching utilities."""

import os
import time
from typing import List

import ccxt
import pandas as pd

from systems.utils.config import resolve_path

COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def _fetch_kraken(symbol: str, start_ms: int, end_ms: int) -> List[List]:
    exchange = ccxt.kraken({"enableRateLimit": True})
    limit = min(720, int((end_ms - start_ms) // 3600000) + 1)
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe="1h", since=start_ms, limit=limit)
    return [row for row in ohlcv if row and start_ms <= row[0] <= end_ms]


def _fetch_binance(symbol: str, start_ms: int, end_ms: int) -> List[List]:
    """Forward pagination with simple retries."""

    exchange = ccxt.binance({"enableRateLimit": True})
    rows: List[List] = []
    limit = 1000
    page_idx = 0
    verbose = os.environ.get("FETCH_DEBUG", "0") == "1"
    MAX_EMPTY_RETRY = 3
    PARTIAL_RETRY_ON_LEN_LT = limit
    SLEEP_ON_RETRY_SEC = 0.5

    since = start_ms
    last_progress_ts: int | None = None

    def _vprint(msg: str) -> None:
        if verbose:
            print(msg)

    while since <= end_ms:
        page = exchange.fetch_ohlcv(symbol, timeframe="1h", since=since, limit=limit) or []
        page_len = len(page)

        empty_retries = 0
        while page_len == 0 and empty_retries < MAX_EMPTY_RETRY:
            _vprint(
                f"[FETCH][BINANCE] page={page_idx} since={since} EMPTY (retry {empty_retries+1}/{MAX_EMPTY_RETRY})"
            )
            time.sleep(SLEEP_ON_RETRY_SEC)
            page = exchange.fetch_ohlcv(symbol, timeframe="1h", since=since, limit=limit) or []
            page_len = len(page)
            empty_retries += 1

        if page_len == 0:
            _vprint(f"[FETCH][BINANCE] floor or no data at since={since} → stop")
            break

        did_partial_retry = False
        if page_len < PARTIAL_RETRY_ON_LEN_LT:
            _vprint(
                f"[FETCH][BINANCE] page={page_idx} got {page_len}/{limit} rows → partial, retrying once"
            )
            time.sleep(SLEEP_ON_RETRY_SEC)
            page2 = exchange.fetch_ohlcv(symbol, timeframe="1h", since=since, limit=limit) or []
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
            f"[FETCH][BINANCE] page={page_idx:05d} since={since} first={first_ts} last={last_ts} rows={page_len} total={len(rows)}"
            f"{' (partial→retry improved)' if did_partial_retry else ''}"
        )

        next_since = last_ts + 1
        if last_progress_ts is not None and next_since <= last_progress_ts:
            _vprint(
                f"[FETCH][BINANCE] no progress (next_since={next_since} <= last_progress={last_progress_ts}) → stop"
            )
            break
        last_progress_ts = next_since
        since = next_since

    return rows


def _rows_to_df(rows: List[List], start_ts: int, end_ts: int) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLUMNS)
    if not df.empty:
        df["timestamp"] = (
            pd.to_datetime(df["timestamp"], unit="ms", utc=True).astype("int64") // 1_000_000_000
        )
        df["timestamp"] = (df["timestamp"] // 3600) * 3600
        df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]
        df[COLUMNS] = df[COLUMNS].apply(pd.to_numeric, errors="coerce")
        df = df.dropna(subset=["timestamp"])
    return df


def fetch_candles(symbol: str, start: int, end: int, source: str) -> pd.DataFrame:
    """Fetch candles for ``symbol`` between ``start`` and ``end`` from ``source``.

    Parameters
    ----------
    symbol: str
        Market symbol understood by the exchange (e.g., ``"SOL/USD"``).
    start: int
        Start timestamp in seconds since epoch.
    end: int
        End timestamp in seconds since epoch.
    source: str
        ``"kraken"`` or ``"binance"``.
    """

    src = source.lower()
    if src == "kraken":
        rows = _fetch_kraken(symbol, int(start * 1000), int(end * 1000))
    elif src == "binance":
        rows = _fetch_binance(symbol, int(start * 1000), int(end * 1000))
    else:
        raise ValueError(f"Unknown source '{source}'")
    return _rows_to_df(rows, start, end)


def load_coin_csv(coin: str) -> pd.DataFrame:
    """Load historical candles for ``coin`` from ``data/raw``."""

    root = resolve_path("")
    path = root / "data" / "raw" / f"{coin.upper()}.csv"
    if not path.exists():  # pragma: no cover - file presence check
        raise FileNotFoundError(f"Candle file not found: {path}")
    return pd.read_csv(path)

