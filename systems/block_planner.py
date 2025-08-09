from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd


DURATION_UNITS = {
    "d": 24,               # days
    "w": 7 * 24,           # weeks
    "m": 30 * 24,          # months (30d)
    "y": 365 * 24,         # years (365d)
}


def parse_duration(text: str) -> int:
    """Convert a duration string (e.g. ``"3m"``) into a candle count."""
    match = re.fullmatch(r"(\d+)([dwmy])", text.strip())
    if not match:
        raise ValueError(f"Invalid duration: {text}")

    value = int(match.group(1))
    unit = match.group(2)
    candles = value * DURATION_UNITS[unit]
    if candles < 1:
        raise ValueError("Duration must be at least one candle")
    return candles


def plan_blocks(df: pd.DataFrame, train_len: int, test_len: int, step_len: int) -> List[Dict[str, int]]:
    """Generate walk-forward blocks.

    Parameters
    ----------
    df : pd.DataFrame
        Candle data sorted by timestamp.
    train_len, test_len, step_len : int
        Lengths in candles.
    """
    total = len(df)
    blocks: List[Dict[str, int]] = []

    start = 0
    while start + train_len + test_len <= total:
        train_start_idx = start
        train_end_idx = start + train_len - 1
        test_start_idx = train_end_idx + 1
        test_end_idx = test_start_idx + test_len - 1

        block = {
            "train_start": int(df.loc[train_start_idx, "timestamp"]),
            "train_end": int(df.loc[train_end_idx, "timestamp"]),
            "test_start": int(df.loc[test_start_idx, "timestamp"]),
            "test_end": int(df.loc[test_end_idx, "timestamp"]),
            "train_candles": train_len,
            "test_candles": test_len,
            "train_index_start": train_start_idx,
            "train_index_end": train_end_idx,
            "test_index_start": test_start_idx,
            "test_index_end": test_end_idx,
        }
        blocks.append(block)
        start += step_len

    return blocks
