from __future__ import annotations

"""Ultra-simple slope/pressure-based sell evaluator."""

import math
import os
from typing import Any, Dict, List

from systems.utils.addlog import addlog


# --- tuning constants -----------------------------------------------------

LOOKBACK = int(os.environ.get("WS_LOOKBACK", 50))
UP_TH = 0.02
DOWN_TH = -0.01


def _slope_and_trend(series, t: int) -> tuple[float, str]:
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


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def evaluate_sell(
    ctx: Dict[str, Any],
    t: int,
    series,
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

    slope, trend = _slope_and_trend(series, t)
    price_now = float(series.iloc[t]["close"])

    notes = [n for n in open_notes if n.get("window_name") == window_name]
    N = len(notes)

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

    avg_roi_pos = sum(max(roi(n), 0.0) for n in notes) / N if N else 0.0
    avg_age = sum(age_bars(n) for n in notes) / N if N else 0.0
    N_norm = min(1.0, N / 10.0)
    R_norm = min(1.0, avg_roi_pos / 0.10)
    A_norm = min(1.0, avg_age / 100.0)
    pressure = _clamp01(0.4 * N_norm + 0.4 * R_norm + 0.2 * A_norm)

    cap = int(cfg.get("max_notes_sell_per_candle", 1))

    if trend == "UP":
        addlog(
            f"[MATURE][{window_name} {cfg['window_size']}] trend={trend} slope={slope:.4f} pressure={pressure:.3f} sold=0/{N} cap={cap}",
            verbose_int=1,
            verbose_state=verbose,
        )
        return []

    if trend == "FLAT":
        f_sell = min(0.20, 0.05 + 0.30 * pressure)
    else:  # DOWN
        f_sell = min(1.00, 0.30 + 0.70 * pressure + 0.50 * abs(min(slope, 0.0)))

    want = math.ceil(f_sell * N)
    want = min(want, cap)

    sorted_notes = sorted(
        notes,
        key=lambda n: (roi(n), value_usd(n), age_bars(n)),
        reverse=True,
    )

    selected: List[Dict[str, Any]] = []
    for note in sorted_notes:
        r = roi(note)
        if r < -0.05:
            continue
        selected.append(note)
        if len(selected) >= want:
            break

    addlog(
        f"[MATURE][{window_name} {cfg['window_size']}] trend={trend} slope={slope:.4f} pressure={pressure:.3f} sold={len(selected)}/{N} cap={cap}",
        verbose_int=1,
        verbose_state=verbose,
    )

    return selected

