from __future__ import annotations

"""Utilities to load and fetch historical candle data."""

import time
from pathlib import Path

import pandas as pd

from systems.utils.addlog import addlog
from systems.utils.config import resolve_path
from systems.scripts.fetch_core import (
    COLUMNS,
    fetch_range,
    get_raw_path,
)


def fetch_candles(coin: str) -> pd.DataFrame:
    """Load historical candles for ``coin`` from ``data/raw``."""
    root = resolve_path("")
    path = root / "data" / "raw" / f"{coin.upper()}.csv"
    if not path.exists():  # pragma: no cover - file presence check
        raise FileNotFoundError(f"Candle file not found: {path}")
    return pd.read_csv(path)


def _write_csv(path: Path, df: pd.DataFrame) -> int:
    tmp = path.with_suffix(".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tmp, index=False)
    tmp.replace(path)
    return len(df)


def fetch_all(coin: str, binance_symbol: str) -> int:
    """Fetch the full Binance history for ``coin`` and overwrite its CSV."""
    end_ts = int(time.time() // 3600 * 3600)
    df = fetch_range("binance", binance_symbol, 0, end_ts).sort_values("timestamp")
    path = get_raw_path(coin.upper())
    rows = len(df)
    addlog(
        f"[FETCH][ALL] Overwriting data/raw/{coin.upper()}.csv with full Binance history ({rows} rows)",
        verbose_int=1,
        verbose_state=True,
    )
    if path.exists():
        path.unlink()
    _write_csv(path, df)
    return rows


def fetch_recent(
    coin: str, kraken_symbol: str, binance_symbol: str, hours: int
) -> int:
    """Fetch ``hours`` of recent data and write to ``coin``'s CSV."""
    end_ts = int(time.time() // 3600 * 3600)
    start_ts = end_ts - hours * 3600
    if hours <= 720:
        df = fetch_range("kraken", kraken_symbol, start_ts, end_ts)
    else:
        kraken_start = end_ts - 720 * 3600
        df_k = fetch_range("kraken", kraken_symbol, kraken_start, end_ts)
        df_b = fetch_range("binance", binance_symbol, start_ts, kraken_start)
        df = pd.concat([df_b, df_k], ignore_index=True)
    df = df.sort_values("timestamp")
    path = get_raw_path(coin.upper())
    rows = len(df)
    addlog(
        f"[FETCH][RECENT] {coin.upper()} {hours}h → {rows} rows from Kraken/Binance",
        verbose_int=1,
        verbose_state=True,
    )
    _write_csv(path, df)
    return rows

