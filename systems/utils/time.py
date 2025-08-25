from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone

import pandas as pd


_INTERVAL_RE = re.compile(r'[_\-]((\d+)([smhdw]))(?=\.|_|$)', re.I)

TIMEFRAME_SECONDS = {
    "s": 1,
    "m": 30 * 24 * 3600,
    "h": 3600,
    "d": 86400,
    "w": 604800,
}

INTERVAL_SECONDS = {
    "s": 1,
    "m": 60,
    "h": 3600,
    "d": 86400,
    "w": 604800,
}

# Minimum rows required for indicators that need lookback
_MIN_REQUIRED_ROWS = 368


def parse_cutoff(value: str) -> timedelta:
    """Return a timedelta parsed from strings like '1d' or '2w'."""
    if not value:
        raise ValueError("cutoff value required")
    value = value.strip().lower()
    if len(value) < 2:
        raise ValueError("cutoff must be in format <num><d|w|m|y>")
    try:
        num = int(value[:-1])
    except ValueError as exc:
        raise ValueError("invalid numeric portion for cutoff") from exc
    unit = value[-1]
    if unit == "h":
        return timedelta(hours=num)
    if unit == "d":
        return timedelta(days=num)
    if unit == "w":
        return timedelta(weeks=num)
    if unit == "m":
        return timedelta(days=30 * num)
    if unit == "y":
        return timedelta(days=365 * num)
    raise ValueError("cutoff unit must be one of h,d,w,m,y")


def parse_duration(value: str) -> timedelta:
    """Return a ``timedelta`` parsed from shorthand like ``'1m'`` or ``'7d'``."""
    return parse_cutoff(value)


def parse_relative_time(value: str) -> tuple[float, float]:
    """Return (start_ts, end_ts) parsed from a relative time string."""
    delta = parse_cutoff(value)
    end = datetime.now(tz=timezone.utc).timestamp()
    start = end - delta.total_seconds()
    return start, end

def duration_from_candle_count(candle_count: int, candle_interval_minutes: int = 60) -> str:
    """
    Converts a number of candles into a human-readable time duration.
    Outputs years, months, days, hours (rounded, no zeroes).
    """
    total_minutes = candle_count * candle_interval_minutes
    total_hours = total_minutes // 60

    days, rem_hours = divmod(total_hours, 24)
    years, rem_days = divmod(days, 365)
    months, days = divmod(rem_days, 30)

    parts = []
    if years:
        parts.append(f"{years}y")
    if months:
        parts.append(f"{months}mo")
    if days:
        parts.append(f"{days}d")
    if rem_hours:
        parts.append(f"{rem_hours}h")

    return " ".join(parts)


def parse_timeframe(tf: str):
    if not tf:
        return None
    m = re.match(r"(?i)^\s*(\d+)\s*([smhdw])\s*$", tf)
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
    if delta is None:
        return df
    if "timestamp" in df.columns:
        ts = df["timestamp"]
        ts_max = float(ts.iloc[-1])
        is_ms = ts_max > 1e12
        to_seconds = (ts / 1000.0) if is_ms else ts
        cutoff = (datetime.now(timezone.utc).timestamp() - delta.total_seconds())
        mask = to_seconds >= cutoff
        return df.loc[mask]
    for col in ("datetime", "date", "time"):
        if col in df.columns:
            try:
                dt = pd.to_datetime(df[col], utc=True, errors="coerce")
                cutoff_dt = pd.Timestamp.utcnow() - delta
                mask = dt >= cutoff_dt
                return df.loc[mask]
            except Exception:
                pass
    sec = infer_candle_seconds_from_filename(file_path) or 3600
    need = int(max(_MIN_REQUIRED_ROWS, delta.total_seconds() // sec))
    if need <= 0 or need >= len(df):
        return df
    return df.iloc[-need:]

