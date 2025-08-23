from __future__ import annotations

"""Offline regime definition for full candle DataFrames."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _load_config() -> dict[str, Any]:
    """Return regime settings from ``settings/settings.json``."""
    path = Path("settings/settings.json")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("regime_settings", {})


def define_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """Annotate ``df`` with a ``regime_true`` column."""
    cfg = _load_config()
    window = int(cfg.get("window", 50))
    slope_eps = float(cfg.get("slope_eps", 0.001))
    vol_eps = float(cfg.get("vol_eps", 0.01))

    closes = df["close"]

    def _slope(series: pd.Series) -> float:
        if series.size < 2:
            return 0.0
        x = np.arange(series.size)
        return float(np.polyfit(x, series.values, 1)[0])

    slope = closes.rolling(window, min_periods=1).apply(_slope, raw=False)
    returns = closes.pct_change()
    vol = returns.rolling(window, min_periods=1).std()

    df = df.copy()
    df["regime_true"] = np.where(
        slope > slope_eps,
        "trend_up",
        np.where(
            slope < -slope_eps,
            "trend_down",
            np.where(vol >= vol_eps, "chop", "flat"),
        ),
    )
    return df


__all__ = ["define_regimes"]
