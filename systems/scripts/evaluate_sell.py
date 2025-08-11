from __future__ import annotations

"""Position-based sell evaluation."""

from typing import Dict, Any, List

from systems.scripts.window_utils import get_window_bounds, get_window_position
from systems.utils.addlog import addlog


def evaluate_sell_actions(
    ctx: Dict[str, Any],
    t: int,
    series,
    cfg: Dict[str, Any],
    open_notes: List[Dict[str, Any]],
    runtime_state: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Return a list of notes to sell on this candle."""

    verbose = 0
    if runtime_state:
        verbose = runtime_state.get("verbose", 0)

    win_low, win_high = get_window_bounds(series, t, cfg["window_size"])
    price = float(series.iloc[t]["close"])
    p = get_window_position(price, win_low, win_high)

    maturity_pos = cfg.get("maturity_position", 1.0)
    if p < maturity_pos:
        addlog(
            f"[HOLD] p={p:.3f} < maturity_position={maturity_pos:.3f}",
            verbose_int=3,
            verbose_state=verbose,
        )
        return []

    candidates = [n for n in open_notes if price >= n.get("target_price", float("inf"))]
    if not candidates:
        return []

    candidates.sort(key=lambda n: (price - n["entry_price"]) / n["entry_price"], reverse=True)
    cap = cfg.get("max_notes_sell_per_candle", 1)
    addlog(
        f"[MATURE] k={len(candidates)} candidates (cap={cap} per candle)",
        verbose_int=2,
        verbose_state=verbose,
    )
    selected = candidates[:cap]
    for note in selected:
        roi = (price - note["entry_price"]) / note["entry_price"] * 100
        addlog(
            f"[SELL] note_id={note.get('id', '')} roi={roi:.2f}% target=${note['target_price']:.4f} price=${price:.4f}",
            verbose_int=1,
            verbose_state=verbose,
        )
    return selected
