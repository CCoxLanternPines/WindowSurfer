from __future__ import annotations

"""Ultra-simple slope-based buy evaluator."""

import os
from typing import Dict, Any

from systems.utils.addlog import addlog


# --- tuning constants -----------------------------------------------------

LOOKBACK = int(os.environ.get("WS_LOOKBACK", 50))
UP_TH = 0.02
DOWN_TH = -0.01

MIN_GAP_BARS = 5
FLAT_BUY_MULT = 0.50
UP_BUY_MULT = 1.00
DOWN_BUY_MULT = 0.00


def _slope_and_trend(series, t: int) -> tuple[float, str]:
    """Return slope and trend classification for candle ``t``."""

    close_now = float(series.iloc[t]["close"])
    close_then = float(series.iloc[max(0, t - LOOKBACK)]["close"])
    slope = (close_now - close_then) / max(close_then, 1e-9)
    if slope >= UP_TH:
        trend = "UP"
    elif slope <= DOWN_TH:
        trend = "DOWN"
    else:
        trend = "FLAT"
    return slope, trend


def evaluate_buy(
    ctx: Dict[str, Any],
    t: int,
    series,
    *,
    window_name: str,
    cfg: Dict[str, Any],
    runtime_state: Dict[str, Any],
):
    """Return sizing and metadata for a buy signal in ``window_name``."""

    verbose = runtime_state.get("verbose", 0)

    slope, trend = _slope_and_trend(series, t)

    capital = float(runtime_state.get("capital", 0.0))
    limits = runtime_state.get("limits", {})
    min_sz = float(limits.get("min_note_size", 0.0))
    max_sz = float(limits.get("max_note_usdt", float("inf")))

    base = float(cfg.get("investment_fraction", 0.0))
    mult = {"UP": UP_BUY_MULT, "FLAT": FLAT_BUY_MULT, "DOWN": DOWN_BUY_MULT}[trend]
    size_usd = capital * base * mult
    raw = size_usd
    size_usd = min(size_usd, capital, max_sz)
    if raw != size_usd:
        addlog(
            f"[CLAMP] size=${raw:.2f} → ${size_usd:.2f} (cap=${capital:.2f}, max=${max_sz:.2f})",
            verbose_int=2,
            verbose_state=verbose,
        )

    last_key = f"last_buy_idx::{window_name}"
    last_idx = int(runtime_state.get(last_key, -1))
    cooldown_ok = (t - last_idx) >= MIN_GAP_BARS

    decision: Dict[str, Any] | bool = False
    if (
        trend != "DOWN"
        and cooldown_ok
        and size_usd >= min_sz
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
        f"[{ 'BUY' if decision else 'SKIP' }][{window_name} {cfg['window_size']}] trend={trend} slope={slope:.4f} mult={mult:.2f} size=${size_usd:.2f}",
        verbose_int=1,
        verbose_state=verbose,
    )

    return decision

