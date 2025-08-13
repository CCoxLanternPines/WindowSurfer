from __future__ import annotations

"""Utilities to load historical candle data."""

import pandas as pd

from systems.utils.config import resolve_path


def fetch_candles(coin: str) -> pd.DataFrame:
    """Load historical candles for ``coin`` from ``data/raw``."""
    root = resolve_path("")
    path = root / "data" / "raw" / f"{coin.upper()}.csv"
    if not path.exists():  # pragma: no cover - file presence check
        raise FileNotFoundError(f"Candle file not found: {path}")
    return pd.read_csv(path)
