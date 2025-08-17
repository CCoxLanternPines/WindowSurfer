from __future__ import annotations

"""Buy evaluation using rolling-window features and pressure balances."""

from typing import Any, Dict, Optional, Tuple

from systems.scripts.math.slope_score import classify_slope
from systems.utils.config import load_settings

CONFIG: Dict[str, Any] = load_settings()

# Constants originally defined in ``sim_engine.py``
WINDOW_SIZE = 24
WINDOW_STEP = 2
STRONG_MOVE_THRESHOLD = 0.15
RANGE_MIN = 0.08
VOLUME_SKEW_BIAS = 0.4
FLAT_BAND_DEG = float(CONFIG.get("flat_band_deg", 10.0))
MAX_PRESSURE = 10.0
BUY_TRIGGER = 3.0


def rule_predict(features: Dict[str, float]) -> int:
    slope = features.get("slope", 0.0)
    rng = features.get("range", 0.0)
    slope_cls = classify_slope(slope, FLAT_BAND_DEG)
    if slope_cls == 0:
        return 0
    if rng < RANGE_MIN:
        return 0
    skew = features.get("volume_skew", 0.0)
    if skew > VOLUME_SKEW_BIAS and slope_cls > 0:
        return 1
    if skew < -VOLUME_SKEW_BIAS and slope_cls < 0:
        return -1
    pct = features.get("pct_change", 0.0)
    if pct >= STRONG_MOVE_THRESHOLD:
        return 2
    if pct > 0:
        return 1
    if pct <= -STRONG_MOVE_THRESHOLD:
        return -2
    if pct < 0:
        return -1
    return 0


def _window_features(window: list[Dict[str, Any]], candle: Any) -> Dict[str, float]:
    """Compute feature dictionary for the current rolling ``window``."""

    open_p = float(window[0].get("open", window[0].get("close", 0.0)))
    close_p = float(window[-1].get("close", 0.0))
    high_p = max(float(c.get("high", close_p)) for c in window)
    low_p = min(float(c.get("low", close_p)) for c in window)

    pct_change = (close_p - open_p) / open_p if open_p else 0.0
    rng = (high_p - low_p) / open_p if open_p else 0.0
    volume_skew = (
        (close_p - (high_p + low_p) / 2) / (high_p - low_p)
        if high_p != low_p
        else 0.0
    )
    slope = (
        (close_p - float(window[0].get("close", open_p))) / float(window[0].get("close", open_p))
        if window[0].get("close", open_p)
        else 0.0
    )
    slope_cls = classify_slope(slope, FLAT_BAND_DEG)

    return {
        "open": open_p,
        "close": close_p,
        "pct_change": pct_change,
        "range": rng,
        "volume_skew": volume_skew,
        "slope": slope,
        "slope_cls": slope_cls,
        "timestamp": candle.get("timestamp"),
        "candle_index": candle.get("candle_index"),
    }


def evaluate_buy(
    candle: Any,
    last_features: Optional[Dict[str, float]],
    state: Dict[str, Any],
    viz_ax=None,
) -> Tuple[Dict[str, Any], Optional[Dict[str, float]]]:
    """Update pressure balances and maybe open a new note."""

    pred = rule_predict(last_features or {})

    bp = state.get("buy_pressure", 0.0)
    sp = state.get("sell_pressure", 0.0)
    if pred > 0:  # upward prediction
        bp = min(MAX_PRESSURE, bp + 1)
        sp = max(0.0, sp - 2)
    elif pred < 0:  # downward prediction
        sp = min(MAX_PRESSURE, sp + 1)
        bp = max(0.0, bp - 2)
    else:  # neutral
        slope_cls = classify_slope((last_features or {}).get("slope", 0.0), FLAT_BAND_DEG)
        if slope_cls == 0:
            sp = min(MAX_PRESSURE, sp + 0.5)
            bp = max(0.0, bp - 0.5)
        else:
            bp = max(0.0, bp - 0.5)
            sp = max(0.0, sp - 0.5)
    state["buy_pressure"] = bp
    state["sell_pressure"] = sp

    if bp >= BUY_TRIGGER:
        price = float(candle.get("close", 0.0))
        note = {"entry_price": price, "timestamp": candle.get("timestamp")}
        state.setdefault("open_notes", []).append(note)
        if viz_ax is not None and candle.get("candle_index") is not None:
            viz_ax.scatter(candle["candle_index"], price, color="green", marker="o")
        state["buy_pressure"] = 0.0

    window = state.setdefault("_window", [])
    window.append(candle)
    if len(window) > WINDOW_SIZE:
        window.pop(0)

    features = last_features
    if len(window) == WINDOW_SIZE and (candle.get("candle_index", 0) % WINDOW_STEP == 0):
        features = _window_features(window, candle)

    return state, features
