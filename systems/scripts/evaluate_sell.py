from __future__ import annotations

"""Price-target-based sell evaluation."""

from typing import Any, Dict, List

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
) -> List[Dict[str, Any]]:
    """Return a list of notes to sell in ``window_name`` on this candle."""

    verbose = 0
    if runtime_state:
        verbose = runtime_state.get("verbose", 0)

    price = float(series.iloc[t]["close"])
    win_low, win_high = get_window_bounds(series, t, cfg["window_size"])

    # Notes belonging to this window
    window_notes = [n for n in open_notes if n.get("window_name") == window_name]
    open_count = len(window_notes)

    ledger = ctx.get("ledger") if ctx else None
    closed_count = 0
    if ledger:
        closed_count = sum(
            1 for n in ledger.get_closed_notes() if n.get("window_name") == window_name
        )

    future_targets = [
        n.get("target_price", float("inf"))
        for n in window_notes
        if n.get("target_price", float("inf")) > price
    ]
    next_target = min(future_targets) if future_targets else None

    # Filter notes for the current window that meet target price
    candidates = [
        n
        for n in window_notes
        if price >= n.get("target_price", float("inf"))
        and price >= n.get("entry_price", float("inf"))
    ]

    if not candidates:
        msg = (
            f"[HOLD][{window_name} {cfg['window_size']}] price=${price:.4f} Notes | "
            f"Open={open_count} | Closed={closed_count} | Next="
        )
        if next_target is not None:
            msg += f"${next_target:.4f}"
        else:
            msg += "None"
        addlog(msg, verbose_int=3, verbose_state=verbose)
        return []

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

    return selected

