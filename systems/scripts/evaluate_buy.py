from __future__ import annotations

"""Buy evaluation driven by predictive pressures."""

from math import atan, degrees
from typing import Any, Dict

import numpy as np

from systems.utils.addlog import addlog
from systems.utils.settings_loader import load_coin_settings


"""Buy evaluation driven by predictive pressures."""


# ---------------------------------------------------------------------------
# Feature extraction and prediction rules
# ---------------------------------------------------------------------------


def classify_slope(slope: float, cfg: Dict[str, Any]) -> int:
    """Return -1 for down, 0 for flat, +1 for up."""
    flat_band_deg = float(cfg["flat_band_deg"])
    angle = degrees(atan(slope))
    if -flat_band_deg <= angle <= flat_band_deg:
        return 0
    return 1 if angle > flat_band_deg else -1


def compute_window_features(series, start: int, cfg: Dict[str, Any]) -> Dict[str, float]:
    """Compute window statistics matching reference logic."""
    window_size = int(cfg["window_size"])
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

    slope_cls = classify_slope(slope, cfg)
    if slope_cls == 0:
        return 0
    if rng < cfg["range_min"]:
        return 0

    skew = features.get("volume_skew", 0.0)
    skew_bias = cfg["volume_skew_bias"]
    if skew > skew_bias and slope_cls > 0:
        return 1
    if skew < -skew_bias and slope_cls < 0:
        return -1

    pct = features.get("pct_change", 0.0)
    strong = cfg["strong_move_threshold"]
    if pct >= strong:
        return 2
    if pct > 0:
        return 1
    if pct <= -strong:
        return -2
    if pct < 0:
        return -1
    return 0


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

def evaluate_buy(
    ctx: Dict[str, Any],
    t: int,
    series,
    *,
    cfg: Dict[str, Any] | None = None,
    runtime_state: Dict[str, Any],
):
    """Return sizing and metadata for a buy signal."""

    if cfg is None:
        market = runtime_state.get("kraken_name") or runtime_state.get("market", "")
        cfg = load_coin_settings(market)

    window_name = "strategy"
    strategy = cfg
    window_size = int(strategy["window_size"])
    window_step = int(strategy["window_step"])

    verbose = runtime_state.get("verbose", 0)

    pressures = runtime_state.setdefault("pressures", {"buy": {}, "sell": {}})

    # Compute features for this window and store for other components
    features = compute_window_features(series, t, strategy)
    runtime_state.setdefault("last_features", {})[window_name] = features

    buy_p = pressures["buy"].get(window_name, 0.0)
    sell_p = pressures["sell"].get(window_name, 0.0)
    max_p = strategy["max_pressure"]

    pred = rule_predict(features, strategy)
    slope_cls = classify_slope(features.get("slope", 0.0), strategy)

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
    if verbose >= 2:
        addlog(
            f"[PRESSURE][{window_name}] buy={buy_p:.1f} sell={sell_p:.1f} pred={pred} slope_cls={slope_cls}",
            verbose_int=2,
            verbose_state=verbose,
        )

    buy_trigger = strategy["buy_trigger"]

    if buy_p < buy_trigger:
        if verbose >= 1:
            addlog(
                f"[HOLD][BUY {window_size}h] need={buy_trigger:.2f}, have={buy_p:.2f}, sell_p={sell_p:.2f}",
                verbose_int=1,
                verbose_state=verbose,
            )
        return False

    fraction = buy_p / max_p if max_p else 0.0
    aggressiveness = strategy.get("buy_percent_aggressiveness", 1.0)
    fraction *= aggressiveness
    fraction = min(fraction, 1.0)

    capital = runtime_state.get("capital", 0.0)
    max_sz = float(strategy["max_note_usdt"])
    min_sz = float(strategy["min_note_size"])

    inv_frac = strategy["investment_fraction"]
    raw = capital * fraction * inv_frac
    size_usd = min(raw, capital, max_sz)
    if size_usd != raw:
        addlog(
            f"[CLAMP] size=${raw:.2f} → ${size_usd:.2f} (cap=${capital:.2f}, max=${max_sz:.2f})",
            verbose_int=2,
            verbose_state=verbose,
        )
    if size_usd < min_sz:
        addlog(
            f"[SKIP][{window_name} {window_size}] size=${size_usd:.2f} < min=${min_sz:.2f}",
            verbose_int=2,
            verbose_state=verbose,
        )
        return False

    addlog(
        f"[BUY][{window_name} {window_size}] pressure={buy_p:.1f}/{max_p:.1f} "
        f"buy_frac={fraction:.2f} agg={aggressiveness:.2f} spend=${size_usd:.2f}",
        verbose_int=1,
        verbose_state=verbose,
    )

    pressures["buy"][window_name] = 0.0

    result = {
        "size_usd": size_usd,
        "window_name": window_name,
        "window_size": window_size,
        "p_buy": fraction,
        "unlock_p": None,
    }
    candle = series.iloc[t]
    if "timestamp" in series.columns:
        result["created_ts"] = int(candle.get("timestamp"))
    result["created_idx"] = t

    return result
