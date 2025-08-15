from __future__ import annotations

"""Buy evaluator with pluggable strategies."""

# ==== PUBLIC VARS (SELF-CONTAINED BUY) ====
# Per-window base sizing (fraction of available capital).
# If a window is not listed, fall back to BUY_BASE_FRACTION_DEFAULT.
BUY_BASE_FRACTION_DEFAULT = 0.08
BUY_BASE_FRACTION_BY_WINDOW = {
    "minnow": 0.08,  # 12h
    "fish":   0.16,  # 2d
}

# Global note size limits (USD) — ignore settings.json
MIN_NOTE_SIZE_USD = 10.0
MAX_NOTE_USD      = 200.0

# Optional cash floor: skip buys if capital below this
CASH_FLOOR_USD    = 350.0

# Slope gating (simple strat)
LOOKBACK     = 50
UP_TH        = +0.025
DOWN_TH      = -0.005
FLAT_BUY_MULT= 0.20
UP_BUY_MULT  = 0.45
MIN_GAP_BARS = 24
# ==========================================

# ==== PUBLIC VARS (choose the buy strategy) ====
BUY_STRAT = "slope"      # options: "slope", "pressure"
# pressure knobs
BUY_WEIGHTS = {"3m": 1.0, "1m": 1.0, "2w": 0.8, "1w": 0.8, "3d": 0.6, "1d": 0.6}
BUY_THRESHOLD_FLAT = 0.50  # min buy pressure in flat
BUY_THRESHOLD_UP   = 0.30  # min buy pressure in up
# ===============================================

from typing import Any, Dict
import pandas as pd

from systems.utils.addlog import addlog

# ---------------------------------------------------------------------------
# Helpers

def trend_from_slope(
    series: pd.DataFrame,
    t: int,
    lookback: int,
    up_th: float,
    down_th: float,
) -> tuple[float, str]:
    """Return (slope, trend) based on ``lookback`` bars."""
    idx = max(0, t - lookback)
    close_then = float(series.iloc[idx]["close"])
    close_now = float(series.iloc[t]["close"])
    slope = (close_now - close_then) / max(close_then, 1e-9)
    if slope >= up_th:
        trend = "UP"
    elif slope <= down_th:
        trend = "DOWN"
    else:
        trend = "FLAT"
    return slope, trend

def _span_to_bars(series: pd.DataFrame, span: str) -> int:
    """Convert a timespan string like '1w' to bar count for ``series``."""
    if not span:
        return 1
    try:
        value = int(span[:-1])
    except ValueError:
        return 1
    unit = span[-1].lower()
    if "timestamp" in series.columns and len(series) >= 2:
        step = int(series.iloc[1]["timestamp"]) - int(series.iloc[0]["timestamp"])
        if step <= 0:
            step = 3600
    else:
        step = 3600
    if unit == "m":
        factor = 30 * 24 * 3600
    elif unit == "w":
        factor = 7 * 24 * 3600
    elif unit == "d":
        factor = 24 * 3600
    elif unit == "h":
        factor = 3600
    else:
        factor = 0
    bars = int(round((value * factor) / step))
    return max(1, bars)

def _window_pos(series: pd.DataFrame, t: int, span: str) -> float:
    """Return position within the span: negative = below center, positive above."""
    bars = _span_to_bars(series, span)
    start = max(0, t - bars + 1)
    window = series.iloc[start : t + 1]
    low = float(window["close"].min())
    high = float(window["close"].max())
    center = 0.5 * (low + high)
    width = max(high - low, 1e-9)
    price_now = float(series.iloc[t]["close"])
    return (price_now - center) / width

# ---------------------------------------------------------------------------
# Strategy router

def evaluate_buy(
    ctx: Dict[str, Any],
    t: int,
    series: pd.DataFrame,
    *,
    window_name: str,
    cfg: Dict[str, Any],
    runtime_state: Dict[str, Any],
):
    """Return sizing and metadata for a buy signal in ``window_name``."""

    verbose = runtime_state.get("verbose", 0)
    capital = float(runtime_state.get("capital", 0.0))

    price_now = float(series.iloc[t]["close"])
    slope, trend = trend_from_slope(series, t, LOOKBACK, UP_TH, DOWN_TH)
    base_frac = BUY_BASE_FRACTION_BY_WINDOW.get(window_name, BUY_BASE_FRACTION_DEFAULT)

    last_key = f"last_buy_idx::{window_name}"
    last_idx = int(runtime_state.get(last_key, -1))
    cooldown_ok = (t - last_idx) >= MIN_GAP_BARS

    decision: Dict[str, Any] | bool = False
    size_usd = 0.0
    score = slope
    label = "slope"
    mult = 0.0

    if BUY_STRAT == "slope":
        if trend != "DOWN":
            mult = FLAT_BUY_MULT if trend == "FLAT" else UP_BUY_MULT
    elif BUY_STRAT == "pressure":
        bp = 0.0
        total_w = 0.0
        for span, weight in BUY_WEIGHTS.items():
            pos = _window_pos(series, t, span)
            s = max(0.0, -pos)
            bp += weight * s
            total_w += weight
        BP = bp / total_w if total_w else 0.0
        label = "bp"
        score = BP
        if trend != "DOWN":
            if (trend == "FLAT" and BP >= BUY_THRESHOLD_FLAT) or (
                trend == "UP" and BP >= BUY_THRESHOLD_UP
            ):
                mult = BP
        # if DOWN or thresholds not met, mult remains 0
    else:
        raise ValueError(f"unknown BUY_STRAT {BUY_STRAT}")

    raw = capital * base_frac * mult
    size_usd = min(raw, MAX_NOTE_USD, capital)
    if raw != size_usd and raw > 0:
        addlog(
            f"[CLAMP] size=${raw:.2f} → ${size_usd:.2f} (cap=${capital:.2f}, max=${MAX_NOTE_USD:.2f})",
            verbose_int=2,
            verbose_state=verbose,
        )

    if (
        cooldown_ok
        and capital >= CASH_FLOOR_USD
        and size_usd >= MIN_NOTE_SIZE_USD
        and size_usd > 0.0
    ):
        decision = {
            "size_usd": size_usd,
            "window_name": window_name,
            "window_size": cfg["window_size"],
            "created_idx": t,
        }
        if "timestamp" in series.columns:
            decision["created_ts"] = int(series.iloc[t]["timestamp"])
        runtime_state[last_key] = t

    addlog(
        f"[{ 'BUY' if decision else 'SKIP' }][{window_name} {cfg['window_size']}] strat={BUY_STRAT} slope={slope:.3f} {label}={score:.3f} trend={trend} size=${size_usd:.2f}",
        verbose_int=1,
        verbose_state=verbose,
    )

    return decision
