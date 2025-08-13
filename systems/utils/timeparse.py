from __future__ import annotations

"""Utility for parsing duration strings to hours."""

from typing import Union
import re


def parse_duration_to_hours(value: Union[str, int, float]) -> int:
    """Return duration in whole hours from various formats.

    * If ``value`` is an int or float, it is interpreted as hours.
    * If ``value`` is a string, recognised suffixes are ``d`` (days),
      ``h`` (hours) and ``m`` (minutes). For example ``"7d"`` -> 168.
    """

    if isinstance(value, (int, float)):
        return int(value)

    if not isinstance(value, str):
        raise TypeError("value must be str, int or float")

    match = re.fullmatch(r"\s*(\d+)([dhm])\s*", value.lower())
    if not match:
        raise ValueError("Invalid duration string")

    amount, unit = match.groups()
    num = int(amount)
    if unit == "d":
        return num * 24
    if unit == "h":
        return num
    if unit == "m":
        # minutes to hours, truncated
        return num // 60
    raise ValueError("Unknown duration unit")
