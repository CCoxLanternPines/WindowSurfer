from __future__ import annotations

"""Sell helper for the simulation engine."""

from typing import Dict, List, Tuple

from systems.scripts.ledger import Ledger
from systems.scripts.window_position_tools import get_trade_params


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
    to_close: List[Dict] = []
    roi_skipped = 0
    for note in ledger.get_active_notes():
        if note["window"] != name:
            continue
        gain_pct = (price - note["entry_price"]) / note["entry_price"]
        trade = get_trade_params(
            price, wave["ceiling"], wave["floor"], cfg, entry_price=note["entry_price"]
        )
        maturity_roi = trade["maturity_roi"]
        if maturity_roi is not None and gain_pct < maturity_roi:
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
