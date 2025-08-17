from __future__ import annotations

"""Window feature extraction, rule-based predictor, and pressure updates."""

from typing import Dict

import numpy as np

from systems.utils.addlog import addlog

try:  # pragma: no cover - optional dependency
    from systems.scripts.math.slope_score import classify_slope  # type: ignore
except Exception:  # pragma: no cover - fallback
    import math

    def classify_slope(slope: float, flat_band_deg: float = 10.0) -> int:
        """Return -1 for down, 0 for flat, +1 for up."""
        angle = math.degrees(math.atan(slope))
        if -flat_band_deg <= angle <= flat_band_deg:
            return 0
        return 1 if angle > flat_band_deg else -1


def compute_window_features(series, start: int, window_size: int) -> Dict[str, float]:
    """Compute window statistics matching reference logic."""
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
    """Classify next window move with multi-feature rules."""
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


def update_pressures(
    state: Dict,
    window_name: str,
    pred: int,
    slope_cls: int,
    cfg: Dict[str, float],
) -> None:
    """Mutate buy/sell pressures based on prediction and slope."""
    pressures = state.setdefault("pressures", {"buy": {}, "sell": {}})
    buy_p = pressures.setdefault("buy", {}).get(window_name, 0.0)
    sell_p = pressures.setdefault("sell", {}).get(window_name, 0.0)
    max_p = cfg.get("max_pressure", 0.0)

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

    pressures["buy"][window_name] = buy_p
    pressures["sell"][window_name] = sell_p

    verbose = state.get("verbose", 0)
    if verbose >= 2:
        addlog(
            f"[PRESSURE][{window_name}] buy={buy_p:.1f} sell={sell_p:.1f} pred={pred} slope_cls={slope_cls}",
            verbose_int=2,
            verbose_state=verbose,
        )

