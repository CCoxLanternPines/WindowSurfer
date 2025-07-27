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
