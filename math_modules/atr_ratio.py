"""ATR(5) over ATR(20) ratio."""

from __future__ import annotations

import pandas as pd

NAME = "atr_ratio"
LOOKBACK = 20


def calculate(df: pd.DataFrame, i: int) -> dict[str, float | None]:
    atr5 = df["atr_5"].iloc[i] if "atr_5" in df.columns else None
    atr20 = df["atr_20"].iloc[i] if "atr_20" in df.columns else None
    if atr5 is None or atr20 in (0, None):
        return {"atr_ratio": None}
    return {"atr_ratio": float(atr5 / atr20) if atr20 else None}
