from __future__ import annotations

"""Sell helper for the simulation engine."""

from typing import Dict, List, Tuple

from systems.scripts.ledger import Ledger


def evaluate_sell(
    *,
    ledger: Ledger,
    name: str,
    tick: int,
    price: float,
    cfg: Dict,
    sim_capital: float,
    verbose: int,
) -> Tuple[float, List[Dict], int]:
    """Close notes that have reached their target price.

    Returns updated capital, the list of notes closed at this tick, and the
    number of notes that failed the minimum ROI requirement.
    """
    min_roi = cfg.get("min_roi", 0)
    to_close: List[Dict] = []
    roi_skipped = 0
    for note in ledger.get_active_notes():
        if note["window"] != name:
            continue
        actual_roi = (price - note["entry_price"]) / note["entry_price"]
        if actual_roi < min_roi:
            if not note.get("min_roi_blocked"):
                roi_skipped += 1
                note["min_roi_blocked"] = True
            continue
        if price >= note["mature_price"]:
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
