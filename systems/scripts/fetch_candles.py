from __future__ import annotations

"""Utilities for loading coin-based candle data."""

import pandas as pd

from systems.utils.config import resolve_path


def load_coin_csv(coin: str) -> pd.DataFrame:
    """Load historical candles for ``coin`` from ``data/raw``.

    Parameters
    ----------
    coin: str
        Base currency ticker (e.g., ``"SOL"``).

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the candle history.

    Raises
    ------
    FileNotFoundError
        If the raw CSV file does not exist.
    """

    root = resolve_path("")
    path = root / "data" / "raw" / f"{coin.upper()}.csv"
    if not path.exists():  # pragma: no cover - file presence check
        raise FileNotFoundError(f"Candle file not found: {path}")
    return pd.read_csv(path)

