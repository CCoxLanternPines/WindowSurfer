from __future__ import annotations

"""Legacy simulation loader with simple time-window filtering.

This module loads CSV candle data for a given symbol and optionally
applies a lookback window relative to the dataset's maximum timestamp.
The ``--time`` parameter specifies the window using whole-day units
(D, W, M, Y) and does **not** alter the filename that is loaded.
"""

from datetime import datetime, timedelta
import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
CANDLES_DIR = DATA_DIR / "candles" / "sim"

_TIME_RE = re.compile(r"^[1-9]\d*[DWMY]$")
_UNIT_DAYS = {"D": 1, "W": 7, "M": 30, "Y": 365}


def _format_market(symbol: str) -> str:
    """Return a display-friendly market string like ``DOGE/USD``."""
    s = symbol.upper()
    if s.endswith("USDC") or s.endswith("USDT"):
        return f"{s[:-4]}/{s[-4:]}"
    return f"{s[:-3]}/{s[-3:]}"


def _parse_window(value: str | None) -> timedelta | None:
    """Parse ``--time`` values like ``1W`` or ``3M`` into ``timedelta``."""
    if value is None:
        return None
    if not _TIME_RE.fullmatch(value):
        print(f"[ERROR] Bad --time value: \"{value}\" (expected e.g. 7D | 2W | 3M | 1Y)")
        raise SystemExit(1)
    num = int(value[:-1])
    unit = value[-1]
    days = num * _UNIT_DAYS[unit]
    return timedelta(days=days)


def load(symbol: str, window: str | None = None) -> pd.DataFrame:
    """Load candle data for ``symbol`` with optional lookback ``window``.

    Parameters
    ----------
    symbol:
        Market pair such as ``DOGEUSD``.
    window:
        Lookback window string like ``1W`` or ``3M``. If omitted, the full
        dataset is returned.
    """

    path = CANDLES_DIR / f"{symbol}.csv"
    abs_path = path.resolve()
    if not path.exists():
        print(f"[ERROR] Missing candles: {abs_path}")
        raise SystemExit(1)

    df = pd.read_csv(path)
    rows_in = len(df)

    if "timestamp" not in df.columns:
        raise ValueError("CSV missing 'timestamp' column")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    t_max = int(df["timestamp"].max()) if rows_in else 0
    t_max_dt = datetime.utcfromtimestamp(t_max)

    delta = _parse_window(window)
    cutoff_ts = t_max
    if delta is not None:
        cutoff_ts = t_max - int(delta.total_seconds())
        df = df[df["timestamp"] >= cutoff_ts]

    rows_out = len(df)
    cutoff_dt = datetime.utcfromtimestamp(cutoff_ts)
    market = _format_market(symbol)
    window_str = window if window is not None else "full"
    print(
        "[SIM][LEGACY] market="
        f"{market} window={window_str} rows_in={rows_in} rows_out={rows_out} "
        f"t_max={t_max_dt.strftime('%Y-%m-%dT%H:%M:%S')} "
        f"cutoff={cutoff_dt.strftime('%Y-%m-%dT%H:%M:%S')}"
    )
    return df.reset_index(drop=True)


__all__ = ["load"]
