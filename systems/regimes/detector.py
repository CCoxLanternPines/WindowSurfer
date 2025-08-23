from __future__ import annotations

"""Online regime detection using only past data."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _load_config() -> dict[str, Any]:
    """Return detector settings from ``settings/settings.json``."""
    path = Path("settings/settings.json")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("regime_settings", {}).get("detector", {})


def detect_regime(df: pd.DataFrame) -> str:
    """Return regime guess for the last candle in ``df``."""
    cfg = _load_config()
    slope_window = int(cfg.get("slope_window", 30))
    vol_window = int(cfg.get("vol_window", 30))
    slope_eps = float(cfg.get("slope_eps", 0.0012))
    vol_eps = float(cfg.get("vol_eps", 0.010))

    closes = df["close"].tail(slope_window)
    if len(closes) < 2:
        slope = 0.0
    else:
        x = np.arange(len(closes))
        slope = float(np.polyfit(x, closes.values, 1)[0])

    closes_for_vol = df["close"].tail(vol_window)
    returns = closes_for_vol.pct_change()
    vol = float(returns.std()) if returns.count() > 1 else 0.0

    if slope > slope_eps:
        return "trend_up"
    if slope < -slope_eps:
        return "trend_down"
    if abs(slope) <= slope_eps and vol >= vol_eps:
        return "chop"
    return "flat"


__all__ = ["detect_regime"]
