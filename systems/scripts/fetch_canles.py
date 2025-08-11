from __future__ import annotations

"""Utilities to load historical candle data."""

import pandas as pd

from systems.utils.config import resolve_path
from systems.utils.addlog import addlog


def fetch_candles(coin: str, fiat: str | None = None) -> pd.DataFrame:
    """Load historical candles for ``coin`` from ``data/raw``."""
    root = resolve_path("")
    path = root / "data" / "raw" / f"{coin.upper()}.csv"
    if not path.exists() and fiat:
        legacy = root / "data" / "raw" / f"{(coin + fiat).upper()}.csv"
        if legacy.exists():
            addlog(
                f"[COMPAT] Using legacy raw file: {legacy.name}",
                verbose_int=1,
                verbose_state=0,
            )
            path = legacy
    if not path.exists():  # pragma: no cover - file presence check
        raise FileNotFoundError(f"Candle file not found: {path}")
    return pd.read_csv(path)
