from __future__ import annotations

"""Utilities to load historical candle data."""

import pandas as pd

from systems.utils.path import find_project_root


def fetch_candles(tag: str) -> pd.DataFrame:
    """Load historical candles for ``tag`` from ``data/raw``."""
    root = find_project_root()
    path = root / "data" / "raw" / f"{tag.upper()}.csv"
    if not path.exists():  # pragma: no cover - file presence check
        raise FileNotFoundError(f"Candle file not found: {path}")
    return pd.read_csv(path)
