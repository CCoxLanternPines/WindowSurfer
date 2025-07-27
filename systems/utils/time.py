from __future__ import annotations

from datetime import datetime, timedelta, timezone


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

