import numpy as np
from pathlib import Path
import csv
from datetime import datetime, timedelta

DATA_DIR = Path(__file__).resolve().parent.parent / 'legacy' / 'data' / 'raw'


def load_prices(tag: str) -> np.ndarray:
    """Load close prices for a given symbol tag.

    Prices are expected to be stored as CSV files under ``legacy/data/raw``
    where the filename matches the asset tag (e.g. ``SOL.csv``).
    Only the ``close`` column is used and returned as a numpy ``float`` array.
    """
    path_csv = DATA_DIR / f"{tag.split('US')[0]}.csv"
    if not path_csv.exists():
        raise FileNotFoundError(f"price data not found for {tag}: {path_csv}")
    closes: list[float] = []
    with path_csv.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            closes.append(float(row['close']))
    return np.array(closes, dtype=float)


def slice_prices(prices: np.ndarray, start: int, end: int) -> np.ndarray:
    """Return a slice of ``prices`` between ``start`` and ``end`` indices."""
    return prices[start:end]


def parse_window(window: str, candles_per_day: int = 1) -> int:
    """Convert a duration like ``'30d'`` or ``'4w'`` into a candle count."""
    if window.isdigit():
        return int(window)
    qty = int(window[:-1])
    unit = window[-1].lower()
    if unit == 'd':
        return qty * candles_per_day
    if unit == 'w':
        return qty * candles_per_day * 7
    if unit == 'm':
        return qty * candles_per_day * 30
    raise ValueError(f"unrecognised window spec: {window}")
