from __future__ import annotations

"""Price-target-based sell evaluation."""

from typing import Any, Dict, List

from systems.scripts.strategy_pressure import (
    pressure_sell_signal,
    pressure_flat_sell_signal,
)
from systems.utils.addlog import addlog


def evaluate_sell(
    ctx: Dict[str, Any],
    t: int,
    series,
    *,
    open_notes: List[Dict[str, Any]],
    runtime_state: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Return a list of notes to sell on this candle."""

    verbose = runtime_state.get("verbose", 0) if runtime_state else 0

    candle = series.iloc[t].to_dict()
    price = float(candle.get("close", 0.0))

    window_notes = [n for n in open_notes if n.get("kind") != "jackpot"]

    selected: List[Dict[str, Any]] = []
    state = runtime_state or {}

    for note in window_notes:
        if pressure_sell_signal(candle, note, state):
            note["action"] = "SELL"
            note["reason"] = "PRESSURE_SELL"
            selected.append(note)
            addlog(
                f"[PRESSURE_SELL] note={note.get('id')} price=${price:.4f}",
                verbose_int=1,
                verbose_state=verbose,
            )

    if pressure_flat_sell_signal(candle, state):
        remaining = [n for n in window_notes if n not in selected]
        if remaining:
            anchor = float(state.get("anchor_price", price))
            drawdown = float(state.get("flat_sell_drawdown", 0.03))
            trigger = anchor * (1.0 - drawdown)
            for note in remaining:
                note["action"] = "SELL"
                note["reason"] = "FLAT_SELL"
                buy = float(note.get("entry_price", 0.0))
                roi = (price - buy) / buy if buy else 0.0
                window_name = note.get("window_name", "")
                addlog(
                    f"[FLAT_SELL][{window_name}] note={note.get('id')} "
                    f"buy={buy:.4f} now={price:.4f} trigger={trigger:.4f} "
                    f"roi={roi*100:.2f}%",
                    verbose_int=1,
                    verbose_state=verbose,
                )
                selected.append(note)

    return selected
