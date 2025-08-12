from __future__ import annotations

"""Price-target-based sell evaluation."""

from typing import Any, Dict, List, Optional

from systems.scripts.window_utils import get_window_bounds
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
) -> Dict[str, Any]:
    """Return sell candidates and next target metadata for ``window_name``.

    Returns a dictionary containing:

    ``notes``
        List of notes selected for sale.
    ``open_notes``
        Count of currently open notes for ``window_name``.
    ``next_sell_price``
        Nearest target price above the current price, if any.
    """

    verbose = 0
    if runtime_state:
        verbose = runtime_state.get("verbose", 0)

    price = float(series.iloc[t]["close"])
    win_low, win_high = get_window_bounds(series, t, cfg["window_size"])

    # Notes belonging to this window
    notes_in_window = [
        n for n in open_notes if n.get("window_name") == window_name
    ]
    open_count = len(notes_in_window)

    # Determine the next target price above the current price, if any
    next_sell_price: Optional[float] = None
    higher_targets = [
        n.get("target_price", float("inf"))
        for n in notes_in_window
        if n.get("target_price", float("inf")) > price
    ]
    if higher_targets:
        next_sell_price = min(higher_targets)

    # Filter notes that have reached their target price
    candidates = [
        n
        for n in notes_in_window
        if price >= n.get("target_price", float("inf"))
        and price >= n.get("entry_price", float("inf"))
    ]

    def roi_now(note: Dict[str, Any]) -> float:
        buy = note.get("entry_price", 0.0)
        return (price - buy) / buy if buy else 0.0

    candidates.sort(key=roi_now, reverse=True)
    cap = cfg.get("max_notes_sell_per_candle", 1)
    selected = candidates[:cap]

    if candidates:
        addlog(
            f"[MATURE][{window_name} {cfg['window_size']}] eligible={len(candidates)} sold={len(selected)} cap={cap}",
            verbose_int=1,
            verbose_state=verbose,
        )
    else:
        msg = (
            f"[HOLD][{window_name} {cfg['window_size']}] price=${price:.4f} "
            f"open_notes={open_count}"
        )
        if next_sell_price is not None:
            msg += f" next_sell=${next_sell_price:.4f}"
        addlog(msg, verbose_int=1, verbose_state=verbose)

    maturity_pos = cfg.get("maturity_position", 1.0)
    for note in selected:
        buy = note.get("entry_price", 0.0)
        qty = note.get("entry_amount", 0.0)
        target = note.get("target_price", 0.0)
        roi = roi_now(note)

        if maturity_pos >= 0.99 and target < buy:
            addlog(
                f"[WARN] SellTargetBelowBuy note=#{note.get('id', '')} buy=${buy:.4f} target=${target:.4f} win=[${win_low:.4f}, ${win_high:.4f}] p_buy={note.get('p_buy', 0.0):.3f}",
                verbose_int=1,
                verbose_state=verbose,
            )
        if roi > 5.0:
            addlog(
                f"[WARN] UnusuallyHighROI note=#{note.get('id', '')} roi={roi*100:.2f}% (check scale/decimals)",
                verbose_int=1,
                verbose_state=verbose,
            )

        addlog(
            f"[SELL][{window_name} {cfg['window_size']}] note=#{note.get('id', '')} qty={qty:.6f} buy=${buy:.4f} now=${price:.4f} target=${target:.4f} roi={roi*100:.2f}%",
            verbose_int=1,
            verbose_state=verbose,
        )

    return {
        "notes": selected,
        "open_notes": open_count,
        "next_sell_price": next_sell_price,
    }

