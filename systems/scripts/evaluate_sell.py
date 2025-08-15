from __future__ import annotations

"""Pressure-based sell evaluator."""

from typing import Any, Dict, List
import math
import pandas as pd

from systems.utils.addlog import addlog


# --- tuning constants -----------------------------------------------------

PRESSURE_WINDOWS = [
    ("3m", 0.20),
    ("1m", 0.20),
    ("2w", 0.15),
    ("1w", 0.15),
    ("3d", 0.15),
    ("1d", 0.15),
]

NORM_N = 10.0
NORM_V = 5_000.0
NORM_A = 100.0
NORM_R = 0.10

LOSS_CAP = -0.05


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


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


def _window_scores(series: pd.DataFrame, t: int) -> Dict[str, Dict[str, float]]:
    scores: Dict[str, Dict[str, float]] = {}
    close_now = float(series.iloc[t]["close"])
    for span, _ in PRESSURE_WINDOWS:
        bars = _span_to_bars(series, span)
        start = max(0, t - bars + 1)
        window = series.iloc[start : t + 1]
        low = float(window["close"].min())
        high = float(window["close"].max())
        width = max(high - low, 1e-9)
        depth = (high - close_now) / width
        height = (close_now - low) / width
        scores[span] = {"depth": depth, "height": height}
    return scores


def _compute_pressures(series: pd.DataFrame, t: int) -> tuple[float, Dict[str, Dict[str, float]]]:
    scores = _window_scores(series, t)
    sp = 0.0
    for span, weight in PRESSURE_WINDOWS:
        sp += weight * scores.get(span, {}).get("height", 0.0)
    return sp, scores


def _compute_sp_load(notes, price_now: float, t: int) -> float:
    N = len(notes)
    val = 0.0
    ages = []
    rois = []
    for n in notes:
        qty = n.get("entry_amount")
        if qty is not None:
            val += float(qty) * price_now
        else:
            val += float(n.get("entry_usd", 0.0))
        created = n.get("created_idx", t)
        ages.append(t - created)
        buy = n.get("entry_price", 0.0)
        if buy:
            rois.append(max((price_now - buy) / buy, 0.0))
    avg_age = sum(ages) / N if N else 0.0
    avg_roi_pos = sum(rois) / N if N else 0.0
    N_norm = min(1.0, N / NORM_N)
    V_norm = min(1.0, val / NORM_V)
    A_norm = min(1.0, avg_age / NORM_A)
    R_norm = min(1.0, avg_roi_pos / NORM_R)
    load = 0.35 * N_norm + 0.35 * V_norm + 0.15 * A_norm + 0.15 * R_norm
    return _clamp01(load)


def _blend_slope(series: pd.DataFrame, t: int) -> float:
    def _slope(bars: int) -> float:
        idx = max(0, t - bars)
        past = float(series.iloc[idx]["close"])
        now = float(series.iloc[t]["close"])
        return (now - past) / max(past, 1e-9)

    b1 = _span_to_bars(series, "1d")
    b3 = _span_to_bars(series, "3d")
    s1 = _slope(b1)
    s3 = _slope(b3)
    return 0.5 * (s1 + s3)


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

    sp_price, scores = _compute_pressures(series, t)
    price_now = float(series.iloc[t]["close"])

    notes = [n for n in open_notes if n.get("window_name") == window_name]
    sp_load = _compute_sp_load(notes, price_now, t)
    sp_total = _clamp01(0.6 * sp_price + 0.4 * sp_load)

    N = len(notes)
    cap = int(cfg.get("max_notes_sell_per_candle", 1))

    slope = _blend_slope(series, t)
    trend_nudge = 0.0
    if slope < 0:
        trend_nudge = 0.10 * abs(slope)

    f_base = 0.02 + 0.28 * sp_total
    f_sell = _clamp01(f_base + trend_nudge)

    want = math.ceil(f_sell * N)
    want = min(want, cap)

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

    selected: List[Dict[str, Any]] = []
    for note in sorted_notes:
        r = roi(note)
        if r < LOSS_CAP:
            continue
        if r < 0 and (sp_total < 0.75 or len(selected) >= want):
            continue
        selected.append(note)
        if len(selected) >= want:
            break

    addlog(
        f"[MATURE][{window_name} {cfg['window_size']}] sp={sp_total:.3f} sold={len(selected)}/{N} cap={cap}",
        verbose_int=1,
        verbose_state=verbose,
    )

    return selected

