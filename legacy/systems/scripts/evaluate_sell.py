from __future__ import annotations

"""Price-target-based sell evaluation."""

from typing import Any, Dict, List

from systems.scripts.window_utils import check_sell_conditions
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

    candle = series.iloc[t]
    price = float(candle["close"])

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

    state = {
        "sell_count": 0,
        "verbose": verbose,
        "window_name": window_name,
        "window_size": cfg["window_size"],
        "max_sells": cfg.get("max_notes_sell_per_candle", 1),
    }

    selected: List[Dict[str, Any]] = []
    for note in candidates:
        if check_sell_conditions(candle, note, cfg, state):
            selected.append(note)

    if candidates:
        addlog(
            f"[MATURE][{window_name} {cfg['window_size']}] eligible={len(candidates)} sold={len(selected)} cap={state['max_sells']}",
            verbose_int=1,
            verbose_state=verbose,
        )

    return selected
