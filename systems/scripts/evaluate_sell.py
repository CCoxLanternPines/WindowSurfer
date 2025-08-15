from __future__ import annotations

"""Simple sell evaluator for discovery mode."""

from typing import Any, Dict, List
import math
import pandas as pd

from systems.utils.addlog import addlog
from .evaluate_buy import LOOKBACK, UP_TH, DOWN_TH

# ==== PUBLIC VARS (discovery mode) ====
LOSS_CAP = -0.05         # max loss allowed on sell (ROI)
FLAT_SELL_FLOOR = 0.02   # 2% of notes per candle in flat trend
FLAT_SELL_MAX = 0.12     # up to 12% if pressure high
DOWN_SELL_BASE = 0.25    # min fraction in downtrend
DOWN_SELL_MAX = 0.60     # max fraction in strong downtrend
# =======================================


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

    verbose = runtime_state.get("verbose", 0) if runtime_state else 0

    price = float(series.iloc[t]["close"])
    idx_then = max(0, t - LOOKBACK)
    price_then = float(series.iloc[idx_then]["close"])
    slope = (price - price_then) / price_then if price_then else 0.0

    if slope >= UP_TH:
        trend = "UP"
    elif slope <= DOWN_TH:
        trend = "DOWN"
    else:
        trend = "FLAT"

    notes = [n for n in open_notes if n.get("window_name") == window_name]
    N = len(notes)
    cap = int(cfg.get("max_notes_sell_per_candle", 1))

    if trend == "UP":
        f_sell = 0.0
    elif trend == "FLAT":
        pressure = min(1.0, N / 10.0)
        f_sell = min(
            FLAT_SELL_FLOOR + pressure * (FLAT_SELL_MAX - FLAT_SELL_FLOOR),
            FLAT_SELL_MAX,
        )
    else:  # DOWN
        f_sell = DOWN_SELL_BASE + abs(slope) * (DOWN_SELL_MAX - DOWN_SELL_BASE)
        f_sell = max(DOWN_SELL_BASE, min(DOWN_SELL_MAX, f_sell))

    want = min(math.ceil(f_sell * N), cap)

    def roi(note: Dict[str, Any]) -> float:
        buy = note.get("entry_price", 0.0)
        return (price - buy) / buy if buy else 0.0

    def value_usd(note: Dict[str, Any]) -> float:
        qty = note.get("entry_amount")
        if qty is not None:
            return float(qty) * price
        return float(note.get("entry_usd", 0.0))

    def age(note: Dict[str, Any]) -> int:
        return t - note.get("created_idx", t)

    sorted_notes = sorted(
        notes,
        key=lambda n: (roi(n), value_usd(n), age(n)),
        reverse=True,
    )

    selected: List[Dict[str, Any]] = []
    for note in sorted_notes:
        r = roi(note)
        if r < LOSS_CAP:
            continue
        if len(selected) >= want:
            break
        selected.append(note)

    top_roi = roi(sorted_notes[0]) if sorted_notes else 0.0
    min_roi_sel = min((roi(n) for n in selected), default=0.0)

    addlog(
        f"[SELL?][{window_name} {cfg['window_size']}] t={t} px={price:.4f} slope={slope:.4f} trend={trend} N={N} f_sell={f_sell:.2f} want={want} cap={cap} selected={len(selected)}/{N} loss_cap={LOSS_CAP} top_roi={top_roi:.3f} min_roi_sel={min_roi_sel:.3f}",
        verbose_int=1,
        verbose_state=verbose,
    )

    return [] if trend == "UP" else selected
