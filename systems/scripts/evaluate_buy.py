from __future__ import annotations

"""Mini-bot compatible buy evaluation."""

from math import atan, degrees
from typing import Any, Dict

import numpy as np


def classify_slope(slope: float, flat_band_deg: float = 10.0) -> int:
    """Return -1 for down, 0 for flat, +1 for up."""
    angle = degrees(atan(slope))
    if -flat_band_deg <= angle <= flat_band_deg:
        return 0
    return 1 if angle > flat_band_deg else -1


def compute_window_features(series, start: int, window_size: int) -> Dict[str, float]:
    """Compute rolling window statistics used by the mini-bot."""
    end = start + window_size
    sub = series.iloc[start:end]

    closes = sub["close"].values
    x = np.arange(len(closes))
    slope = float(np.polyfit(x, closes, 1)[0]) if len(closes) > 1 else 0.0
    volatility = float(np.std(closes)) if len(closes) else 0.0

    low = float(sub["low"].min()) if "low" in sub else float(sub["close"].min())
    high = float(sub["high"].max()) if "high" in sub else float(sub["close"].max())
    rng = high - low

    vol_mean = float(sub["volume"].mean()) if "volume" in sub else 0.0
    mid = len(sub) // 2
    if mid and "volume" in sub:
        early = float(sub["volume"].iloc[:mid].mean())
        late = float(sub["volume"].iloc[mid:].mean())
        volume_skew = ((late - early) / early) if early else 0.0
    else:
        volume_skew = 0.0

    level = float(sub.iloc[0]["close"]) if len(sub) else 0.0
    exit_price = float(sub.iloc[-1]["close"]) if len(sub) else 0.0
    pct_change = (exit_price - level) / level if level else 0.0

    return {
        "slope": slope,
        "volatility": volatility,
        "range": rng,
        "volume_mean": vol_mean,
        "volume_skew": volume_skew,
        "pct_change": pct_change,
    }


def rule_predict(features: Dict[str, float], cfg: Dict[str, float]) -> int:
    """Classify the next window move with mini-bot rules."""
    slope = features.get("slope", 0.0)
    rng = features.get("range", 0.0)

    slope_cls = classify_slope(slope, cfg.get("flat_band_deg", 10.0))
    if slope_cls == 0:
        return 0
    if rng < cfg.get("range_min", 0.0):
        return 0

    skew = features.get("volume_skew", 0.0)
    skew_bias = cfg.get("volume_skew_bias", 0.0)
    if skew > skew_bias and slope_cls > 0:
        return 1
    if skew < -skew_bias and slope_cls < 0:
        return -1

    pct = features.get("pct_change", 0.0)
    strong = cfg.get("strong_move_threshold", 0.0)
    if pct >= strong:
        return 2
    if pct > 0:
        return 1
    if pct <= -strong:
        return -2
    if pct < 0:
        return -1
    return 0


def evaluate_buy(
    t: int,
    series,
    *,
    cfg: Dict[str, float],
    state: Dict[str, Any],
) -> Dict[str, float] | None:
    """Update pressures and return buy size if triggered."""

    window_size = int(cfg.get("window_size", 0))
    start = t + 1 - window_size
    if start < 0:
        state["last_features"] = None
        return None

    last_features = state.get("last_features")
    buy_p = state.get("buy_pressure", 0.0)
    sell_p = state.get("sell_pressure", 0.0)
    max_p = float(cfg.get("max_pressure", 7.0))

    if last_features is not None:
        pred = rule_predict(last_features, cfg)
        slope_cls = classify_slope(
            last_features.get("slope", 0.0), cfg.get("flat_band_deg", 10.0)
        )
        if pred > 0:
            buy_p = min(max_p, buy_p + 1)
            sell_p = max(0.0, sell_p - 2)
        elif pred < 0:
            sell_p = min(max_p, sell_p + 1)
            buy_p = max(0.0, buy_p - 2)
        else:
            if slope_cls == 0:
                sell_p = min(max_p, sell_p + 0.5)
                buy_p = max(0.0, buy_p - 0.5)
            else:
                buy_p = max(0.0, buy_p - 0.5)
                sell_p = max(0.0, sell_p - 0.5)

    features = compute_window_features(series, start, window_size)
    state["last_features"] = features
    state["buy_pressure"] = buy_p
    state["sell_pressure"] = sell_p

    if buy_p >= cfg.get("buy_trigger", 0.0):
        trade_size = buy_p / max_p if max_p else 0.0
        state["buy_pressure"] = 0.0
        return {"trade_size": trade_size}
    return None

