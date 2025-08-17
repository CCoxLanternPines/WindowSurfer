from __future__ import annotations

"""Helper for parsing timeframe strings into timedeltas."""

from datetime import timedelta

from systems.utils.time import parse_cutoff


def parse_timeframe(value: str) -> timedelta | None:
    """Parse a timeframe string like ``'7d'`` into a ``timedelta``.

    Returns ``None`` if ``value`` is falsy or invalid.
    """
    if not value:
        return None
    try:
        return parse_cutoff(value)
    except Exception:
        return None
