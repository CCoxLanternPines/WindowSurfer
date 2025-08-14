from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Tuple

import pandas as pd

from systems.scripts.fetch_candles import fetch_kraken_range
from systems.utils.resolve_symbol import (
    to_tag as _to_tag,
    live_path_csv,
    sim_path_csv,
)

_HIST_CACHE: dict[str, Tuple[float, float]] = {}


def to_tag(symbol: str) -> str:
    """Return uppercase tag without separators for exchange symbol."""
    return _to_tag(symbol)


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


def refresh_live_kraken_720(symbol: str) -> None:
    """Refresh last 720h from Kraken into live cache with locking."""
    tag = to_tag(symbol)
    end_ts = last_closed_hour_ts(int(time.time()))
    path = live_path_csv(tag)
    dir_path = os.path.dirname(path)
    os.makedirs(dir_path, exist_ok=True)
    lock_path = os.path.join(dir_path, f"{tag}.refresh.lock")
    meta_path = os.path.join(dir_path, f"{tag}.meta.json")

    # Check meta for up-to-date
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            if meta.get("last_refresh_ts") == end_ts:
                print("[LIVE][REFRESH][SKIP] reason=up_to_date")
                return
        except Exception:
            pass

    now = int(time.time())
    # Lock handling
    if os.path.exists(lock_path):
        mtime = os.path.getmtime(lock_path)
        if now - mtime < 180:
            print("[LIVE][REFRESH][SKIP] reason=locked")
            return
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass

    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        print("[LIVE][REFRESH][SKIP] reason=locked")
        return

    try:
        df = fetch_kraken_range_1h(symbol, end_ts, n=720)
        rows = len(df)
        iso_end = datetime.fromtimestamp(end_ts, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:00Z"
        )
        print(
            f"[LIVE][REFRESH] kraken symbol={symbol} tag={tag} end={iso_end} n={rows}"
        )
        tmp = path + ".tmp"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
        print(f"[LIVE][REFRESH] wrote={rows} path={path}")
        if rows < 720:
            print(f"[LIVE][REFRESH][WARN] exchange returned {rows} bars (<720)")
        with open(meta_path, "w") as f:
            json.dump({"last_refresh_ts": end_ts, "last_closed_ts": end_ts}, f)
    finally:
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass


__all__ = [
    "to_tag",
    "live_path_csv",
    "sim_path_csv",
    "load_sim_for_high_low",
    "last_closed_hour_ts",
    "fetch_kraken_range_1h",
    "refresh_live_kraken_720",
]
