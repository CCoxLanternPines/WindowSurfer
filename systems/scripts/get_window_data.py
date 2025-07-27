from __future__ import annotations

import pandas as pd
from systems.utils.path import find_project_root
from systems.utils.time import parse_cutoff


def get_window_data(tag: str, window: str, candle_offset: int = 0, verbose: bool = False) -> dict | None:
    if verbose:
        print(
            f"[get_window_data] tag={tag} window={window} candle_offset={candle_offset}"
        )

    root = find_project_root()
    path = root / "data" / "raw" / f"{tag.upper()}.csv"

    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        if verbose:
            print(f"[ERROR] Data file not found: {path}")
        return None

    if df.empty:
        if verbose:
            print("[WARN] CSV is empty")
        return None

    # Convert window to duration in candles (assume hourly)
    duration = parse_cutoff(window)
    num_candles = int(duration.total_seconds() // 3600)

    # Truncate if not enough candles for full window
    start_idx = max(0, len(df) - candle_offset - num_candles)
    end_idx = len(df) - candle_offset if candle_offset != 0 else None
    window_df = df.iloc[start_idx:end_idx]

    if window_df.empty:
        if verbose:
            print("[WARN] No candle data in computed window slice")
        return None

    ceiling = window_df["high"].max()
    floor = window_df["low"].min()

    try:
        last_candle = df.iloc[-1 - candle_offset]
        close = last_candle["close"]
    except IndexError:
        if verbose:
            print(f"[ERROR] Not enough candles to read offset {candle_offset}")
        return None

    range_val = ceiling - floor
    tunnel_position = (close - floor) / range_val if range_val != 0 else 0.5
    window_position = floor if range_val != 0 else 0.0

    result = {
        "window_ceiling": round(ceiling, 8),
        "window_floor": round(floor, 8),
        "tunnel_position": round(tunnel_position, 4),
        "window_position": round(window_position, 4)
    }

    if verbose:
        print(f"[get_window_data] result={result}")

    return result
