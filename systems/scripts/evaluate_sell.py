from __future__ import annotations

"""Sell signal evaluation used by simulation and live paths."""

from typing import Dict, List

from systems.scripts.ledger import Ledger
from systems.scripts.window_position_tools import get_trade_params
from systems.utils.addlog import addlog


def evaluate_sell(
    *,
    state: Dict,
    ledger: Ledger,
    strategy: str,
    cfg: Dict,
    wave: Dict,
    tick: int,
    price: float,
    cooldowns: Dict[str, Dict[str, int]],
    verbose: int,
) -> List[Dict]:
    """Return a list of sell signals for ``strategy``."""

    cd = cooldowns.get(strategy, {}).get("sell", 0)
    if cd > 0:
        return []

    notes = [
        n
        for n in ledger.get_open_notes()
        if n.get("strategy") == strategy and n.get("status") == "open" and not n.get("closed") and n.get("last_action_tick") != tick
    ]
    notes.sort(
        key=lambda n: (price - n["entry_price"]) / n["entry_price"],
        reverse=True,
    )

    signals: List[Dict] = []
    for note in notes:
        gain_pct = (price - note["entry_price"]) / note["entry_price"]
        trade_note = get_trade_params(
            current_price=price,
            window_high=wave["ceiling"],
            window_low=wave["floor"],
            config=cfg,
            entry_price=note["entry_price"],
        )
        maturity_roi = trade_note["maturity_roi"]
        if maturity_roi is not None:
            addlog(
                f"[DEBUG][SELL] gain_pct={gain_pct:.2%} maturity_roi={maturity_roi:.2%}",
                verbose_int=3,
                verbose_state=verbose,
            )
        else:
            addlog(
                f"[DEBUG][SELL] gain_pct={gain_pct:.2%} maturity_roi=None",
                verbose_int=3,
                verbose_state=verbose,
            )
        if (
            maturity_roi is None
            or maturity_roi <= 0
            or gain_pct < 0
            or gain_pct < maturity_roi
        ):
            continue
        signals.append({"note": note, "price": price})
    return signals
