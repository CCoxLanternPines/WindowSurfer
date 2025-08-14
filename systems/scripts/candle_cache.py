from __future__ import annotations

import os
import time
from typing import Tuple

import pandas as pd

from systems.scripts.fetch_candles import fetch_kraken_range

_HIST_CACHE: dict[str, Tuple[float, float]] = {}


def tag_from_symbol(symbol: str) -> str:
    """Return uppercase tag without slash for exchange symbol."""
    return symbol.replace("/", "").upper()


def live_path_csv(tag: str) -> str:
    return f"data/live/{tag}_1h.csv"


def sim_path_csv(tag: str) -> str:
    return f"data/sim/{tag}_1h.csv"


def load_sim_for_high_low(tag: str) -> Tuple[float, float]:
    """Load historical low/high from SIM cache, cached in-memory."""
    if tag in _HIST_CACHE:
        return _HIST_CACHE[tag]
    path = sim_path_csv(tag)
    df = pd.read_csv(path)
    low = float(df["low"].min())
    high = float(df["high"].max())
    _HIST_CACHE[tag] = (low, high)
    return low, high


def last_closed_hour_ts(now_utc: int) -> int:
    """Return Unix timestamp of the last closed UTC hour."""
    return int((now_utc // 3600 - 1) * 3600)


def fetch_kraken_range_1h(symbol: str, end_ts: int, n: int = 720) -> pd.DataFrame:
    """Fetch N Kraken hourly candles ending at ``end_ts`` inclusive."""
    start_ts = end_ts - (n - 1) * 3600
    return fetch_kraken_range(symbol, start_ts, end_ts)


def hard_refresh_live_720(symbol: str) -> None:
    """Fetch last 720h for ``symbol`` and atomically write live cache."""
    tag = tag_from_symbol(symbol)
    end_ts = last_closed_hour_ts(int(time.time()))
    df = fetch_kraken_range_1h(symbol, end_ts, n=720)
    path = live_path_csv(tag)
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
    rows = len(df)
    if rows < 720:
        print(f"[LIVE][REFRESH][WARN] exchange returned {rows} bars (<720); wrote anyway")
    print(f"[LIVE][REFRESH] fetched={rows} wrote={path} (atomic)")


__all__ = [
    "tag_from_symbol",
    "live_path_csv",
    "sim_path_csv",
    "load_sim_for_high_low",
    "last_closed_hour_ts",
    "fetch_kraken_range_1h",
    "hard_refresh_live_720",
]
