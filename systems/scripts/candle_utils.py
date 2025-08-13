"""Helper utilities for working with candle data."""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Tuple

from systems.utils.config import resolve_path


def compute_ath_atl(df: pd.DataFrame) -> Tuple[float | None, float | None]:
    """Return the all-time high and low for ``df``.

    The dataframe is expected to contain a ``close`` column. ``None`` is
    returned for both values when ``df`` is empty.
    """
    if df is None or df.empty:
        return None, None
    ath = float(df["close"].max())
    atl = float(df["close"].min())
    return ath, atl


def load_full_history(coin: str) -> pd.DataFrame | None:
    """Load a pre-fetched full-history file for ``coin`` if it exists.

    The file is expected at ``data/raw/<COIN>_full.csv`` relative to the
    repository root. ``None`` is returned when the file is missing or could not
    be parsed.
    """
    root = resolve_path("")
    path = root / "data" / "raw" / f"{coin.upper()}_full.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None
