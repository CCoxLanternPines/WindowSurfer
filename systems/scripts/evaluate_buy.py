from __future__ import annotations

"""Simple buy evaluator for discovery mode."""

from typing import Any, Dict
import pandas as pd

from systems.utils.addlog import addlog

# ==== PUBLIC VARS (discovery mode) ====
LOOKBACK = 50           # bars to look back for slope
UP_TH = +0.02           # >= +2% slope = uptrend
DOWN_TH = -0.01         # <= -1% slope = downtrend (halt buys)
FLAT_BUY_MULT = 0.25    # fraction of base size in flat trend
UP_BUY_MULT = 0.60      # fraction of base size in uptrend
MIN_GAP_BARS = 12       # cooldown between buys
# =======================================


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

    price = float(series.iloc[t]["close"])
    idx_then = max(0, t - LOOKBACK)
    price_then = float(series.iloc[idx_then]["close"])
    slope = (price - price_then) / price_then if price_then else 0.0

    if slope >= UP_TH:
        trend = "UP"
        mult = UP_BUY_MULT
    elif slope <= DOWN_TH:
        trend = "DOWN"
        mult = 0.0
    else:
        trend = "FLAT"
        mult = FLAT_BUY_MULT

    capital = float(runtime_state.get("capital", 0.0))
    base = float(cfg.get("investment_fraction", 0.0))
    raw = capital * base * mult
    limits = runtime_state.get("limits", {})
    min_sz = float(limits.get("min_note_size", 0.0))
    max_sz = float(limits.get("max_note_usdt", float("inf")))
    size_clamped = min(raw, capital, max_sz)
    too_small = size_clamped < min_sz or size_clamped <= 0.0
    size_usd = max(min_sz, size_clamped)

    gap_key = f"last_buy_idx::{window_name}"
    last_idx = int(runtime_state.get(gap_key, -1))
    gap_ok = (t - last_idx) >= MIN_GAP_BARS

    decision: Dict[str, Any] | bool = False
    reason = ""
    if trend == "DOWN":
        reason = "trend=DOWN"
    elif not gap_ok:
        reason = "cooldown"
    elif too_small:
        reason = "too_small"
    else:
        reason = "ok"
        decision = {
            "size_usd": size_usd,
            "window_name": window_name,
            "window_size": cfg["window_size"],
            "created_idx": t,
        }
        if "timestamp" in series.columns:
            decision["created_ts"] = int(series.iloc[t]["timestamp"])
        runtime_state[gap_key] = t

    log_msg = (
        f"[BUY?][{window_name} {cfg['window_size']}] t={t} px={price:.4f} "
        f"slope={slope:.4f} trend={trend} mult={mult:.2f} gap_ok={gap_ok} "
        f"size_raw=${raw:.2f} clamp(min={min_sz},max={max_sz})→${size_usd:.2f} "
        f"decision={'BUY' if decision else 'SKIP'} reason={reason}"
    )
    addlog(
        log_msg,
        verbose_int=1,
        verbose_state=verbose,
    )

    return decision
