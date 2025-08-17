from __future__ import annotations

"""Buy evaluation based on pressure and simple feature rules."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

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


def _extract_features(candle: Any, last_close: Optional[float]) -> Dict[str, float]:
    """Compute basic features from the current candle.

    Parameters
    ----------
    candle: Any
        Candle row providing ``open``, ``high``, ``low`` and ``close`` values.
    last_close: Optional[float]
        Close price from the previous candle.
    """

    open_p = float(candle.get("open", 0.0))
    close_p = float(candle.get("close", 0.0))
    high_p = float(candle.get("high", close_p))
    low_p = float(candle.get("low", close_p))

    pct_change = (close_p - open_p) / open_p if open_p else 0.0
    volatility = (high_p - low_p) / open_p if open_p else 0.0
    skew = (close_p - (high_p + low_p) / 2) / (high_p - low_p) if high_p != low_p else 0.0
    slope = 0.0
    if last_close is not None and last_close:
        slope = (close_p - last_close) / last_close
    slope_cls = 1 if slope > STRONG_MOVE_THRESHOLD else -1 if slope < -STRONG_MOVE_THRESHOLD else 0

    return {
        "open": open_p,
        "close": close_p,
        "pct_change": pct_change,
        "volatility": volatility,
        "skew": skew,
        "slope": slope,
        "slope_cls": slope_cls,
        "timestamp": candle.get("timestamp"),
        "candle_index": candle.get("candle_index"),
    }


def _rule_predict(features: Dict[str, float]) -> float:
    """Heuristic rule used to accumulate buy pressure."""

    score = 0.0
    if features["volatility"] > RANGE_MIN:
        score += 1.0
    if features["pct_change"] > 0:
        score += 1.0
    if features["skew"] > VOLUME_SKEW_BIAS:
        score += 1.0
    return score


def evaluate_buy(
    candle: Any,
    last_features: Optional[Dict[str, float]],
    state: Dict[str, Any],
    viz_ax=None,
) -> Tuple[Dict[str, float], Dict[str, Any], Optional[Dict[str, Any]]]:
    """Update buy pressure and possibly open a new position.

    Parameters
    ----------
    candle:
        Current market candle.
    last_features:
        Feature dictionary from previous candle; may be ``None``.
    state:
        Mutable strategy state.
    viz_ax:
        Optional Matplotlib axis for plotting buy markers.
    """

    last_close = last_features.get("close") if last_features else None
    features = _extract_features(candle, last_close)

    pred = _rule_predict(features)
    bp = state.get("buy_pressure", 0.0)
    bp = min(MAX_PRESSURE, bp + pred + features["slope_cls"])
    state["buy_pressure"] = bp

    trade = None
    if bp >= BUY_TRIGGER:
        note = {
            "entry_price": features["close"],
            "timestamp": features.get("timestamp"),
        }
        state.setdefault("open_notes", []).append(note)
        trade = note
        if viz_ax is not None and features.get("candle_index") is not None:
            viz_ax.scatter(features["candle_index"], features["close"], color="green", marker="o")
        state["buy_pressure"] = 0.0

    return features, state, trade
