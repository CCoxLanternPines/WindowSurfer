from __future__ import annotations
from typing import Any, Dict
import numpy as np
from math import atan2
from systems.utils.addlog import addlog

def evaluate_buy(
    ctx: Dict[str, Any],
    t: int,
    series,
    *,
    cfg: Dict[str, Any],
    runtime_state: Dict[str, Any],
) -> Dict[str, Any] | None:
    """
    Source-of-truth buy evaluation.
    Returns a normalized trade dict or None.
    """

    capital = runtime_state.get("capital", 0.0)
    if capital <= 0:
        return None

    candle = series.iloc[t]
    price = float(candle["close"])

    # --- Exhaustion bubble ---
    lookback = cfg.get("exhaustion_lookback", 184)
    if t < lookback:
        return None
    past_price = float(series["close"].iloc[t - lookback])
    if price >= past_price:
        return None
    delta_down = past_price - price
    norm_down = delta_down / max(1e-9, past_price)
    bubble = 1_000_000 * (norm_down ** 3)

    # --- Base size ---
    bmin = cfg.get("buy_min_bubble", 100)
    bmax = cfg.get("buy_max_bubble", 500)
    pmin = cfg.get("min_note_size_pct", 0.03)
    pmax = cfg.get("max_note_size_pct", 0.25)
    if bubble < bmin:
        return None
    frac = min(max((bubble - bmin) / (bmax - bmin), 0.0), 1.0)
    size_usd = capital * (pmin + frac * (pmax - pmin))

    if size_usd <= 0 or size_usd > capital:
        return None

    # --- Trend multiplier ---
    angle_lb = cfg.get("angle_lookback", 48)
    if t >= angle_lb:
        dy = price - float(series["close"].iloc[t - angle_lb])
        dx = angle_lb
        angle = atan2(dy, dx)
        norm = angle / (np.pi / 4)
    else:
        norm = 0.0
    trend_mult = 0.25 + (1.0 - 0.25) * max(-1.0, min(1.0, norm))

    # --- Volatility multiplier ---
    vol_lb = cfg.get("vol_lookback", 48)
    if "returns" not in series:
        series["returns"] = series["close"].pct_change()
    if "volatility" not in series:
        series["volatility"] = series["returns"].rolling(vol_lb).std().fillna(0)
    vol = float(series["volatility"].iloc[t])
    vmin = cfg.get("buy_min_vol_bubble", 0.0)
    vmax = cfg.get("buy_max_vol_bubble", 0.01)
    vol_frac = (min(max(vol, vmin), vmax) - vmin) / max(1e-9, vmax - vmin)
    vol_mult = 2.5 + vol_frac * (0.0 - 2.5)

    # --- Final allocation ---
    trade_usd = size_usd * trend_mult * vol_mult
    if trade_usd <= 0 or trade_usd > capital:
        return None
    units = trade_usd / price

    # --- Sell target ---
    smin = cfg.get("sell_min_bubble", 150)
    smax = cfg.get("sell_max_bubble", 800)
    mmin = cfg.get("min_maturity", 0.05)
    mmax = cfg.get("max_maturity", 0.25)
    s_clamped = min(max(bubble, smin), smax)
    frac = (s_clamped - smin) / (smax - smin)
    maturity = mmin + frac * (mmax - mmin)
    sell_price = price * (1 + maturity)

    addlog(
        f"[BUY] idx={t} price={price:.5f} usd={trade_usd:.2f} units={units:.5f} target={sell_price:.5f}",
        verbose_int=1,
        verbose_state=runtime_state.get("verbose", 0),
    )

    return {
        "side": "BUY",
        "entry_price": price,
        "size_usd": trade_usd,
        "units": units,
        "created_idx": t,
        "created_ts": int(candle.get("timestamp", 0)),
        "sell_price": sell_price,
    }
