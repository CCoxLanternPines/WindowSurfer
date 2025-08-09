"""Basic candle CSV loader for Phase 0 simulation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Dict


def load_candles(tag: str) -> List[Dict[str, float]]:
    """Load candles for the given trading pair tag.

    Expects a CSV file at data/raw/<TAG>.csv with columns:
    timestamp, open, high, low, close, volume
    """
    path = Path("data/raw") / f"{tag}.csv"
    candles: List[Dict[str, float]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            candles.append(
                {
                    "timestamp": row["timestamp"],
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                }
            )
    return candles
