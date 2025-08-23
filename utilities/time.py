from __future__ import annotations

"""Utility helpers for parsing time window strings."""

from datetime import timedelta
import re

_TIME_RE = re.compile(r"^[1-9]\d*[DWMY]$")
_UNIT_DAYS = {"D": 1, "W": 7, "M": 30, "Y": 365}


def parse_time_window(value: str) -> timedelta:
    r"""Return ``timedelta`` parsed from values like ``1W`` or ``3M``.

    Parameters
    ----------
    value:
        A string matching ``^[1-9]\d*[DWMY]$``.

    Raises
    ------
    ValueError
        If ``value`` does not conform to the expected pattern.
    """
    if not _TIME_RE.fullmatch(value):
        raise ValueError("bad time window")
    num = int(value[:-1])
    unit = value[-1]
    days = num * _UNIT_DAYS[unit]
    return timedelta(days=days)


__all__ = ["parse_time_window"]
