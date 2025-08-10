from __future__ import annotations

import argparse
import re
from typing import Optional, Tuple

import pandas as pd


__all__ = ["add_verbosity", "validate_dates", "parse_duration_1h"]


def add_verbosity(parser: argparse.ArgumentParser) -> None:
    """Add a ``-v`` counting flag to ``parser``."""
    parser.add_argument("-v", action="count", default=0, dest="verbosity")


def validate_dates(start: Optional[str], end: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Validate that ``start`` and ``end`` are provided together and ordered."""
    if start and end:
        s = pd.Timestamp(start, tz="UTC")
        e = pd.Timestamp(end, tz="UTC")
        if e <= s:
            raise argparse.ArgumentTypeError("end must be after start")
        return start, end
    if start or end:
        raise argparse.ArgumentTypeError("--start and --end must be provided together")
    return None, None


_DURATION_UNITS = {
    "h": 1,
    "d": 24,
    "w": 7 * 24,
    "m": 30 * 24,
    "y": 365 * 24,
}


def parse_duration_1h(text: str) -> int:
    """Convert duration strings like ``"3m"`` to 1h candle counts."""
    match = re.fullmatch(r"(\d+)([hdwmy])", text.strip())
    if not match:
        raise argparse.ArgumentTypeError(f"Invalid duration: {text}")
    value = int(match.group(1))
    unit = match.group(2)
    candles = value * _DURATION_UNITS[unit]
    if candles < 1:
        raise argparse.ArgumentTypeError("Duration must be positive")
    return candles
