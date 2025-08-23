from __future__ import annotations

import re
from datetime import timedelta, datetime, timezone
import os

import pandas as pd

from .utils.brain_math import (
    WINDOW_SIZE,
    WINDOW_STEP,
    CLUSTER_WINDOW,
    BASE_SIZE,
    SCALE_POWER,
    multi_window_vote,
)

_INTERVAL_RE = re.compile(r'[_\-]((\d+)([smhdw]))(?=\.|_|$)', re.I)

TIMEFRAME_SECONDS = {
    's': 1,
    'm': 30 * 24 * 3600,  # month (≈30 days)
    'h': 3600,
    'd': 86400,
    'w': 604800,
}

INTERVAL_SECONDS = {
    's': 1,
    'm': 60,
    'h': 3600,
    'd': 86400,
    'w': 604800,
}


def parse_timeframe(tf: str) -> timedelta | None:
    """Parse strings like '12h', '3d', '1m', '6w' into timedelta."""
    if not tf:
        return None
    m = re.match(r'(?i)^\s*(\d+)\s*([smhdw])\s*$', tf)
    if not m:
        return None
    n, u = int(m.group(1)), m.group(2).lower()
    return timedelta(seconds=n * TIMEFRAME_SECONDS[u])

def infer_candle_seconds_from_filename(path: str) -> int | None:
    m = _INTERVAL_RE.search(os.path.basename(path))
    if not m:
        return None
    n, u = int(m.group(2)), m.group(3).lower()
    return n * INTERVAL_SECONDS[u]

def apply_time_filter(df: pd.DataFrame, delta: timedelta, file_path: str) -> pd.DataFrame:
    """Filter dataframe to the requested timeframe."""
    if delta is None:
        return df
    if 'timestamp' in df.columns:
        ts = df['timestamp']
        ts_max = float(ts.iloc[-1])
        is_ms = ts_max > 1e12
        to_seconds = (ts / 1000.0) if is_ms else ts
        cutoff = (datetime.now(timezone.utc).timestamp() - delta.total_seconds())
        mask = to_seconds >= cutoff
        return df.loc[mask]
    for col in ('datetime','date','time'):
        if col in df.columns:
            try:
                dt = pd.to_datetime(df[col], utc=True, errors='coerce')
                cutoff_dt = pd.Timestamp.utcnow() - delta
                mask = dt >= cutoff_dt
                return df.loc[mask]
            except Exception:
                pass
    sec = infer_candle_seconds_from_filename(file_path) or 3600
    need = int(delta.total_seconds() // sec)
    if need <= 0 or need >= len(df):
        return df
    return df.iloc[-need:]


def run_simulation(timeframe: str = "1m", viz: bool = True) -> None:
    """Placeholder simulation runner.

    MetaBrain components have been removed. Use ``brain.py`` to audit or teach
    individual brains instead of this aggregated simulator.
    """

    raise RuntimeError(
        "MetaBrain components are disabled. Use brain.py for standalone brains."
    )

