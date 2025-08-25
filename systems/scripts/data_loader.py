from __future__ import annotations

import csv
from collections import deque
from pathlib import Path
from typing import Dict, Any

from systems.utils.config import resolve_path
from systems.utils.addlog import addlog


def _extract_candle_row(df, row_offset: int = 0) -> dict | None:
    """Return a candle row from a dataframe if available."""
    if df is None or df.empty or row_offset >= len(df):
        return None

    row = df.iloc[-(1 + row_offset)]
    return {
        "timestamp": int(row["timestamp"]),
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
        "volume": float(row["volume"]),
    }


def get_candle_data_df(df, row_offset: int = 0) -> dict | None:
    """Return candle data from a preloaded dataframe."""
    try:
        import pandas as pd  # noqa: F401
    except Exception:  # pragma: no cover - pandas may not be installed
        return None

    return _extract_candle_row(df, row_offset)


def get_candle_data_json(tag: str, row_offset: int = 0) -> dict | None:
    """Load candle data from CSV for ``tag`` and return a row."""
    try:
        import pandas as pd
    except Exception:  # pragma: no cover - pandas may not be installed
        pd = None  # type: ignore

    root = resolve_path("")
    path: Path = root / "data" / "raw" / f"{tag.upper()}.csv"

    if pd is None:
        if not path.exists():
            return None
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            last_rows = deque(reader, maxlen=row_offset + 1)
            if len(last_rows) <= row_offset:
                return None
            row = last_rows[-(1 + row_offset)]
            return {
                "timestamp": int(row["timestamp"]),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }

    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return None

    return _extract_candle_row(df, row_offset)


def get_candle_data(tag: str, row_offset: int = 0, verbose: int = 0) -> Dict[str, Any]:
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

    addlog(
        f"[get_candle_data] tag={tag} row_offset={row_offset}",
        verbose_int=3,
        verbose_state=verbose,
    )

    root = resolve_path("")
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

    result = {
        "timestamp": int(row["timestamp"]),
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
        "volume": float(row["volume"]),
    }

    addlog(
        f"[get_candle_data] result={result}",
        verbose_int=3,
        verbose_state=verbose,
    )

    return result


def get_window_data(df, window: str, candle_offset: int = 0) -> dict | None:
    """
    Return window-relative wave structure data for current price.

    Returns:
        dict with:
            - window: str (e.g. "3d")
            - floor: float
            - ceiling: float
            - range: float
            - price: float
            - position_in_window: float (0 = floor, 1 = ceiling)
    """
    from systems.utils.time import parse_cutoff

    if df is None or df.empty:
        return None
    try:
        from systems.utils.time import parse_cutoff
        window_duration = parse_cutoff(window)
        num_candles = int(window_duration.total_seconds() // 3600)

        curr_close = float(df.iloc[len(df) - candle_offset - 1]["close"])
        past_close = float(
            df.iloc[len(df) - candle_offset - num_candles]["close"]
        )
        # Normalize to % change: (current - past) / past
        trend_direction_delta_window = (
            ((curr_close - past_close) / past_close) * 100 if past_close else 0.0
        )
    except (IndexError, KeyError, ZeroDivisionError):
        trend_direction_delta_window = 0.0

    duration = parse_cutoff(window)
    num_candles = int(duration.total_seconds() // 3600)

    start_idx = max(0, len(df) - candle_offset - num_candles)
    end_idx = len(df) - candle_offset if candle_offset != 0 else None
    window_df = df.iloc[start_idx:end_idx]

    if window_df.empty:
        return None

    try:
        last_candle = df.iloc[-1 - candle_offset]
        price = float(last_candle["close"])
    except IndexError:
        return None

    floor = float(window_df["low"].min())
    ceiling = float(window_df["high"].max())
    range_val = ceiling - floor
    position = (price - floor) / range_val if range_val != 0 else 0.5

    return {
        "window": window,
        "floor": round(floor, 6),
        "ceiling": round(ceiling, 6),
        "range": round(range_val, 6),
        "price": round(price, 6),
        "position_in_window": round(position, 4),
        "trend_direction_delta_window": trend_direction_delta_window,
    }
