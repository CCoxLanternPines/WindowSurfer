"""Mean percent change of last 5 closes."""

from __future__ import annotations

import pandas as pd

NAME = "short_slope"
LOOKBACK = 5


def calculate(df: pd.DataFrame, i: int) -> dict[str, float | None]:
    closes = df["close"].iloc[i - 4 : i + 1]
    if len(closes) < 5:
        return {"slope": None}
    pct = closes.pct_change().dropna()
    return {"slope": float(pct.mean())}
