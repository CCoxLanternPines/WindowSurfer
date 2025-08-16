from __future__ import annotations

"""Switchback-based buy evaluator."""

from typing import Any, Dict
import pandas as pd

from systems.scripts.math.trend_switch import slope_stats, classify_trend_z
from systems.utils.addlog import addlog

# ---- Switchback knobs (public; tweak in-file) ----
WINDOW_LOOKBACK = {"minnow": 24, "fish": 48}
Z_HI = 1.5
Z_LO = -1.5
PERSIST_K = {"minnow": 2, "fish": 3}
MIN_ABS_SLOPE = 0.0005
# Buy sizing & gates
BASE_BUY_FRACTION = {"minnow": 0.125, "fish": 0.25}
MIN_NOTE_USD = 10.0
CASH_FLOOR_USD = 0.0
BUY_COOLDOWN_BARS = {"minnow": 2, "fish": 2}
# Sell gates (unused here but kept for parity)
MAX_SELL_PER_CANDLE = {"minnow": 1, "fish": 1}


def _trend_switch(series: pd.DataFrame, t: int, window: str, state: Dict[str, Any]):
    look = WINDOW_LOOKBACK.get(window, 0)
    if t + 1 < look or look <= 0:
        return 0.0, 0.0, "FLAT", 0, "FLAT", False, ""
    prices = series["close"].iloc[t - look + 1 : t + 1].tolist()
    slope, se = slope_stats(prices)
    z = slope / se if se else 0.0
    side = classify_trend_z(slope, se, Z_HI, Z_LO, MIN_ABS_SLOPE)

    trend_state = state.setdefault("trend_state", {})
    ts = trend_state.setdefault(window, {"last_trend": "FLAT", "persist_count": 0, "prev_side": "FLAT"})
    prev_trend = ts["last_trend"]

    if side == ts.get("prev_side") and side in ("UP", "DOWN"):
        ts["persist_count"] += 1
    elif side in ("UP", "DOWN"):
        ts["prev_side"] = side
        ts["persist_count"] = 1
    else:
        ts["prev_side"] = side
        ts["persist_count"] = 0

    switchback = False
    switch_desc = ""
    if side in ("UP", "DOWN") and ts["persist_count"] >= PERSIST_K.get(window, 1):
        confirmed = side
        if prev_trend in ("UP", "DOWN") and confirmed != prev_trend:
            switchback = True
            switch_desc = f"{prev_trend}→{confirmed}"
        ts["last_trend"] = confirmed
    return slope, z, side, ts["persist_count"], prev_trend, switchback, switch_desc


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
    slope, z, side, persist, last_trend, switchback, switch_desc = _trend_switch(
        series, t, window_name, runtime_state
    )
    addlog(
        f"[TREND][{window_name}] slope={slope:.5f} z={z:.2f} side={side} "
        f"persist={persist}/{PERSIST_K.get(window_name, 0)} last={last_trend} switchback={switch_desc}",
        verbose_int=1,
        verbose_state=verbose,
    )

    if switch_desc != "DOWN→UP":
        return False

    cooldowns = runtime_state.setdefault("cooldowns", {}).setdefault("buy", {})
    last_idx = int(cooldowns.get(window_name, -10**9))
    cd = BUY_COOLDOWN_BARS.get(window_name, 0)
    if t - last_idx < cd:
        return False

    cash = float(runtime_state.get("capital", 0.0))
    size_usd = min(cash, BASE_BUY_FRACTION.get(window_name, 0.0) * cash)
    if cash <= CASH_FLOOR_USD or size_usd < MIN_NOTE_USD:
        return False

    decision: Dict[str, Any] = {
        "size_usd": size_usd,
        "window_name": window_name,
        "window_size": cfg.get("window_size"),
        "created_idx": t,
        "reason": f"[BUY][{window_name}] switchback DOWN→UP",
    }
    if "timestamp" in series.columns:
        decision["created_ts"] = int(series.iloc[t]["timestamp"])
    cooldowns[window_name] = t
    addlog(
        f"[BUY][{window_name}] switchback DOWN→UP size=${size_usd:.2f}",
        verbose_int=1,
        verbose_state=verbose,
    )
    return decision
