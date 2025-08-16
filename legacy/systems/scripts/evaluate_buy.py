from __future__ import annotations

"""Pressure-based buy evaluator."""

from typing import Any, Dict
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

MIN_GAP_BARS = 12

NORM_N = 10.0
NORM_V = 5_000.0
NORM_A = 100.0
NORM_R = 0.10

BP_SIG_CENTER = 0.40
BP_SIG_WIDTH = 0.12


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


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


def _compute_pressures(series: pd.DataFrame, t: int) -> tuple[float, float, Dict[str, Dict[str, float]]]:
    scores = _window_scores(series, t)
    bp = 0.0
    sp = 0.0
    for span, weight in PRESSURE_WINDOWS:
        sc = scores.get(span, {})
        bp += weight * sc.get("depth", 0.0)
        sp += weight * sc.get("height", 0.0)
    return bp, sp, scores


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

    bp_price, sp_price, scores = _compute_pressures(series, t)
    close_now = float(series.iloc[t]["close"])

    ledger = ctx.get("ledger")
    notes = []
    if ledger is not None:
        notes = ledger.get_open_notes()
    sp_load = _compute_sp_load(notes, close_now, t)
    sp_total = _clamp01(0.6 * sp_price + 0.4 * sp_load)

    short_height = 0.5 * (
        scores.get("1d", {}).get("height", 0.0)
        + scores.get("3d", {}).get("height", 0.0)
    )

    if short_height > 0.65 and sp_total > 0.6:
        addlog(
            f"[GATE][{window_name} {cfg['window_size']}] short_h={short_height:.3f} sp={sp_total:.3f}",
            verbose_int=2,
            verbose_state=verbose,
        )
        return False

    capital = float(runtime_state.get("capital", 0.0))
    limits = runtime_state.get("limits", {})
    min_sz = float(limits.get("min_note_size", 0.0))
    max_sz = float(limits.get("max_note_usdt", float("inf")))

    base = float(cfg.get("investment_fraction", 0.0))
    m_buy = _sigmoid((bp_price - BP_SIG_CENTER) / BP_SIG_WIDTH)
    size_usd = capital * base * m_buy
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
    if cooldown_ok and size_usd >= min_sz and size_usd > 0.0:
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
        f"[{ 'BUY' if decision else 'SKIP' }][{window_name} {cfg['window_size']}] bp={bp_price:.3f} sp={sp_total:.3f} size=${size_usd:.2f}",
        verbose_int=1,
        verbose_state=verbose,
    )

    return decision

