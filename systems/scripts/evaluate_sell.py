from __future__ import annotations

"""Sell evaluator with pluggable strategies."""

# ==== PUBLIC VARS (SELF-CONTAINED SELL) ====
# Per-window max sells per candle — ignore settings.json
SELL_CAP_PER_CANDLE_DEFAULT = 2
SELL_CAP_PER_CANDLE_BY_WINDOW = {
    "minnow": 2,  # 12h
    "fish":   2,  # 2d
}

# Loss cap: never sell below this ROI (e.g., -5% max loss)
LOSS_CAP = -0.05

# Slope knobs for sell behavior
LOOKBACK_SELL  = 50
UP_TH_SELL     = +0.025
DOWN_TH_SELL   = -0.005

# Flat regime bleed
FLAT_SELL_FLOOR = 0.04
FLAT_SELL_MAX   = 0.15

# Downtrend exit speed
DOWN_SELL_BASE  = 0.35
DOWN_SELL_MAX   = 0.70
# ===========================================

# ==== PUBLIC VARS (choose the sell strategy) ====
SELL_STRAT = "pressure"  # options: "slope", "pressure"
# pressure-mode knobs
SELL_WEIGHTS = {"3m": 1.0, "1m": 1.0, "2w": 0.8, "1w": 0.8, "3d": 0.6, "1d": 0.6}
SELL_THRESHOLD_FLAT = 0.60
SELL_THRESHOLD_DOWN = 0.30
# ===============================================

from typing import Any, Dict, List
import math
import pandas as pd

from systems.utils.addlog import addlog
from systems.scripts.evaluate_buy import trend_from_slope

# ---------------------------------------------------------------------------
# Helpers

def _span_to_bars(series: pd.DataFrame, span: str) -> int:
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

    verbose = 0
    if runtime_state:
        verbose = runtime_state.get("verbose", 0)

    notes = [n for n in open_notes if n.get("window_name") == window_name]
    N = len(notes)
    if N == 0:
        return []

    price_now = float(series.iloc[t]["close"])
    slope, trend = trend_from_slope(series, t, LOOKBACK_SELL, UP_TH_SELL, DOWN_TH_SELL)
    cap = SELL_CAP_PER_CANDLE_BY_WINDOW.get(window_name, SELL_CAP_PER_CANDLE_DEFAULT)
    score = slope
    label = "slope"

    selected: List[Dict[str, Any]] = []

    if SELL_STRAT == "slope":
        if trend == "UP":
            want = 0
        else:
            pressure = min(1.0, N / 10.0)
            if trend == "FLAT":
                f_sell = min(
                    FLAT_SELL_MAX,
                    FLAT_SELL_FLOOR + pressure * (FLAT_SELL_MAX - FLAT_SELL_FLOOR),
                )
            else:  # DOWN
                f_sell = DOWN_SELL_BASE + abs(slope) * (DOWN_SELL_MAX - DOWN_SELL_BASE)
                f_sell = max(DOWN_SELL_BASE, min(DOWN_SELL_MAX, f_sell))
            want = max(0, min(N, math.ceil(f_sell * N)))
            want = min(want, cap)
    elif SELL_STRAT == "pressure":
        sp = 0.0
        total_w = 0.0
        for span, weight in SELL_WEIGHTS.items():
            pos = _window_pos(series, t, span)
            sp += weight * max(0.0, pos)
            total_w += weight
        SP_price = sp / total_w if total_w else 0.0
        load = min(1.0, N / 10.0)
        SP = max(0.0, min(1.0, 0.7 * SP_price + 0.3 * load))
        label = "sp"
        score = SP
        gate_ok = True
        if trend == "UP":
            gate_ok = False
        elif trend == "FLAT" and SP < SELL_THRESHOLD_FLAT:
            gate_ok = False
        elif trend == "DOWN" and SP < SELL_THRESHOLD_DOWN:
            gate_ok = False
        if not gate_ok:
            want = 0
        else:
            f_sell = SP
            if trend == "FLAT":
                f_sell = min(SP, FLAT_SELL_MAX)
            want = max(0, min(N, math.ceil(f_sell * N)))
            want = min(want, cap)
    else:
        raise ValueError(f"unknown SELL_STRAT {SELL_STRAT}")

    def roi(note: Dict[str, Any]) -> float:
        buy = note.get("entry_price", 0.0)
        return (price_now - buy) / buy if buy else 0.0

    def age_bars(note: Dict[str, Any]) -> int:
        return t - note.get("created_idx", t)

    def value_usd(note: Dict[str, Any]) -> float:
        qty = note.get("entry_amount")
        if qty is not None:
            return float(qty) * price_now
        return float(note.get("entry_usd", 0.0))

    sorted_notes = sorted(
        notes,
        key=lambda n: (roi(n), value_usd(n), age_bars(n)),
        reverse=True,
    )

    for note in sorted_notes:
        if roi(note) < LOSS_CAP:
            continue
        if len(selected) >= want:
            break
        selected.append(note)

    sold = len(selected)
    addlog(
        f"[MATURE][{window_name} {cfg['window_size']}] strat={SELL_STRAT} {label}={score:.3f} trend={trend} sold={sold}/{N} cap={cap}",
        verbose_int=1,
        verbose_state=verbose,
    )

    return selected
