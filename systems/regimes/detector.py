from __future__ import annotations

"""Online regime detection using only past data."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _load_config() -> dict[str, Any]:
    path = Path("settings/settings.json")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("regime_settings", {}).get("detector", {})


def detect_regime(df: pd.DataFrame) -> str:
    """Return regime guess for the last candle in ``df``."""
    cfg = _load_config()
    slope_window = int(cfg.get("slope_window", 30))
    vol_window = int(cfg.get("vol_window", 30))
    slope_eps = float(cfg.get("slope_eps", 0.001))
    vol_eps = float(cfg.get("vol_eps", 0.01))
    wick_eps = float(cfg.get("wick_eps", 0.003))
    range_eps = float(cfg.get("range_eps", 0.02))
    consistency_up = float(cfg.get("consistency_up", 0.65))
    consistency_down = float(cfg.get("consistency_down", 0.35))

    closes = df["close"].tail(slope_window)
    if len(closes) < 2:
        slope = 0.0
    else:
        x = np.arange(len(closes))
        slope = float(np.polyfit(x, closes.values, 1)[0])

    window = df.tail(vol_window)
    returns = window["close"].pct_change()
    vol = float(returns.std())
    rng = (window["high"].max() - window["low"].min()) / window["close"].mean()
    wick = (
        (window["high"] - window[["open", "close"]].max(axis=1))
        + (window[["open", "close"]].min(axis=1) - window["low"])
    ).mean() / window["close"].mean()
    consistency = (window["close"] > window["open"]).mean()

    if slope > slope_eps and consistency >= consistency_up:
        regime = "trend_up"
    elif slope < -slope_eps and consistency <= consistency_down:
        regime = "trend_down"
    elif vol > vol_eps and (wick > wick_eps or rng > range_eps):
        regime = "chop"
    else:
        regime = "flat"
    return regime


__all__ = ["detect_regime"]
