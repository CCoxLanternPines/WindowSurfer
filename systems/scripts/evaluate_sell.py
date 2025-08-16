from __future__ import annotations

"""Switchback-based sell evaluator."""

from typing import Any, Dict, List
import pandas as pd

from systems.scripts.math.trend_switch import slope_stats, classify_trend_z
from systems.utils.addlog import addlog

# ---- Switchback knobs (public; tweak in-file) ----
WINDOW_LOOKBACK = {"minnow": 24, "fish": 48}
Z_HI = 1.5
Z_LO = -1.5
PERSIST_K = {"minnow": 2, "fish": 3}
MIN_ABS_SLOPE = 0.0005
# Buy sizing & gates (unused here but kept for parity)
BASE_BUY_FRACTION = {"minnow": 0.125, "fish": 0.25}
MIN_NOTE_USD = 10.0
CASH_FLOOR_USD = 0.0
BUY_COOLDOWN_BARS = {"minnow": 2, "fish": 2}
# Sell gates
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


def evaluate_sell(
    ctx: Dict[str, Any],
    t: int,
    series: pd.DataFrame,
    *,
    window_name: str,
    cfg: Dict[str, Any],
    open_notes: List[Dict[str, Any]],
    runtime_state: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Return a list of notes to sell in ``window_name`` on this candle."""

    runtime_state = runtime_state or {}
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

    if switch_desc != "UP→DOWN":
        return []

    notes = [n for n in open_notes if n.get("window_name") == window_name]
    limit = MAX_SELL_PER_CANDLE.get(window_name, 0)
    selected = notes[:limit]
    for n in selected:
        n["reason"] = f"[SELL][{window_name}] switchback UP→DOWN"
    addlog(
        f"[SELL][{window_name}] switchback UP→DOWN count={len(selected)}",
        verbose_int=1,
        verbose_state=verbose,
    )
    return selected
