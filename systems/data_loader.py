from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


CACHE_DIR = Path("data/raw")


def load_candles(tag: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
    """Load cached 1h candles for *tag*.

    Parameters
    ----------
    tag : str
        Symbol or asset identifier. A CSV named ``<tag>.csv`` is expected
        in the cache directory.
    start_date, end_date : str | None, optional
        ISO8601 timestamp strings (any ``pandas.Timestamp`` compatible format).
        When provided, the resulting frame is filtered to the inclusive
        range ``[start_date, end_date]``.

    Returns
    -------
    pd.DataFrame
        Dataframe with columns ``timestamp, open, high, low, close, volume``
        sorted in ascending timestamp order with duplicate timestamps removed.
    """

    path = CACHE_DIR / f"{tag}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No cached data for tag '{tag}' at {path}")

    df = pd.read_csv(path, usecols=["timestamp", "open", "high", "low", "close", "volume"])

    # Sort and deduplicate timestamps
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)

    # Ensure strictly 1h interval candles
    if not df.empty:
        gaps = df["timestamp"].diff().dropna()
        if not (gaps == 3600).all():
            raise ValueError("Non 1h candle intervals detected in cached data")

    # Optional time filtering
    if start_date is not None:
        start_ts = int(pd.Timestamp(start_date, tz="UTC").timestamp())
        df = df[df["timestamp"] >= start_ts]
    if end_date is not None:
        end_ts = int(pd.Timestamp(end_date, tz="UTC").timestamp())
        df = df[df["timestamp"] <= end_ts]

    return df.reset_index(drop=True)
