from __future__ import annotations

"""Sell helper for the simulation engine."""

from typing import Dict, List, Tuple

from systems.scripts.ledger import Ledger
from systems.scripts.window_position_tools import get_trade_params
from systems.utils.addlog import addlog


def evaluate_sell(
    *,
    ledger: Ledger,
    name: str,
    tick: int,
    price: float,
    wave: Dict,
    cfg: Dict,
    sim_capital: float,
    verbose: int,
) -> Tuple[float, List[Dict], int]:
    """Close notes that have reached their target price.

    Returns updated capital, the list of notes closed at this tick, and the
    number of notes that failed the minimum ROI requirement.
    """
    trade = get_trade_params(price, wave["ceiling"], wave["floor"], cfg)
    if trade["in_dead_zone"]:
        return sim_capital, [], 0

    to_close: List[Dict] = []
    roi_skipped = 0
    for note in ledger.get_active_notes():
        if note["window"] != name:
            continue
        gain_pct = (price - note["entry_price"]) / note["entry_price"]
        trade_note = get_trade_params(
            price, wave["ceiling"], wave["floor"], cfg, entry_price=note["entry_price"]
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
        if maturity_roi is None or maturity_roi <= 0 or gain_pct < maturity_roi:
            roi_skipped += 1
            continue
        gain = (price - note["entry_price"]) * note["entry_amount"]
        note["exit_tick"] = tick
        note["exit_price"] = price
        note["exit_ts"] = tick
        note["gain"] = gain
        base = note["entry_price"] * note["entry_amount"] or 1
        note["gain_pct"] = gain / base
        note["status"] = "Closed"
        to_close.append(note)
    closed: List[Dict] = []
    for note in to_close:
        ledger.close_note(note)
        sim_capital += note["entry_amount"] * price
        closed.append(note)
    return sim_capital, closed, roi_skipped
