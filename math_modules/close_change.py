"""Example metric: difference between current and previous close."""

from __future__ import annotations

import pandas as pd


def calculate(df: pd.DataFrame, idx: int) -> dict[str, float]:
    """Return the change in closing price from previous candle."""
    if idx <= 0 or idx >= len(df):
        return {"close_change": float("nan")}
    prev_close = df.loc[idx - 1, "close"]
    curr_close = df.loc[idx, "close"]
    return {"close_change": curr_close - prev_close}
