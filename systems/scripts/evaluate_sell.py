from __future__ import annotations

"""Position-based sell evaluation."""

from typing import Dict, Any, List

from systems.scripts.window_utils import get_window_bounds, get_window_position
from systems.utils.addlog import addlog


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

    win_low, win_high = get_window_bounds(series, t, cfg["window_size"])
    price = float(series.iloc[t]["close"])
    p = get_window_position(price, win_low, win_high)

    maturity_pos = cfg.get("maturity_position", 1.0)
    if p < maturity_pos:
        addlog(
            f"[HOLD][{window_name} {cfg['window_size']}] p={p:.3f} < maturity={maturity_pos:.2f}",
            verbose_int=3,
            verbose_state=verbose,
        )
        return []

    candidates = [
        n
        for n in open_notes
        if n.get("window_name") == window_name and price >= n.get("target_price", float("inf"))
    ]
    if not candidates:
        return []

    candidates.sort(
        key=lambda n: (price - n["entry_price"]) / n["entry_price"], reverse=True
    )
    cap = cfg.get("max_notes_sell_per_candle", 1)
    selected = candidates[:cap]
    for note in selected:
        roi = (price - note["entry_price"]) / note["entry_price"]
        addlog(
            f"[SELL][{window_name} {cfg['window_size']}] note={note.get('id', '')} roi={roi*100:.2f}% target=${note['target_price']:.4f} price=${price:.4f}",
            verbose_int=1,
            verbose_state=verbose,
        )
    return selected
