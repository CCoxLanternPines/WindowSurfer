from __future__ import annotations

import csv
from collections import deque
from pathlib import Path
from typing import Dict, Any

from systems.utils.path import find_project_root


def get_candle_data(tag: str, row_offset: int = 0) -> Dict[str, Any]:
    """Return the most recent candle for ``tag`` from the raw CSV data.

    Parameters
    ----------
    tag : str
        Market or ticker tag (e.g. ``"DOGEUSD"``). Case-insensitive.
    row_offset : int, optional
        Offset from the latest row. ``0`` selects the most recent candle,
        ``1`` selects the candle before that, and so on.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing ``timestamp``, ``open``, ``high``, ``low``,
        ``close`` and ``volume`` as numeric types.

    Raises
    ------
    FileNotFoundError
        If the CSV file for ``tag`` does not exist.
    IndexError
        If the requested row does not exist in the file.
    """

    root = find_project_root()
    path: Path = root / "data" / "raw" / f"{tag.upper()}.csv"

    if not path.exists():
        raise FileNotFoundError(f"Raw candle file not found: {path}")

    # Try to use pandas for convenience if available
    row = None
    try:
        import pandas as pd
    except Exception:  # pragma: no cover - pandas may not be installed
        pd = None  # type: ignore

    if pd is not None:
        df = pd.read_csv(path)
        if row_offset >= len(df):
            raise IndexError(
                f"File {path} contains only {len(df)} rows, cannot access offset {row_offset}"
            )
        row = df.iloc[-(1 + row_offset)].to_dict()
    else:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            last_rows = deque(reader, maxlen=row_offset + 1)
            if len(last_rows) <= row_offset:
                raise IndexError(
                    f"File {path} does not contain row with offset {row_offset}"
                )
            row = last_rows[-(1 + row_offset)]

    return {
        "timestamp": int(row["timestamp"]),
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
        "volume": float(row["volume"]),
    }
