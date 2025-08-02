from __future__ import annotations

"""Core buy and sell helpers for the simulation engine."""

from typing import Dict

from systems.scripts.ledger import Ledger
from systems.utils.addlog import addlog


def maybe_buy(
    *,
    ledger: Ledger,
    name: str,
    cfg: Dict,
    wave: Dict,
    tick: int,
    price: float,
    sim_capital: float,
    last_buy_tick: Dict[str, int],
    max_note_usdt: float,
    min_note_usdt: float,
    verbose: int,
) -> float:
    """Attempt to open a new note if buy conditions are met."""
    position = wave["position_in_window"]
    if position <= cfg.get("buy_floor", 0):
        cooldown = cfg.get("cooldown", 0)
        if tick - last_buy_tick.get(name, float("-inf")) >= cooldown:
            open_for_window = [n for n in ledger.get_active_notes() if n["window"] == name]
            if len(open_for_window) < cfg.get("max_open_notes", 0):
                invest = sim_capital * cfg.get("investment_fraction", 0)
                invest = min(invest, max_note_usdt)
                if invest >= min_note_usdt and invest <= sim_capital:
                    amount = invest / price
                    mature_price = wave["floor"] + wave["range"] * cfg.get("sell_ceiling", 1)
                    note = {
                        "window": name,
                        "entry_tick": tick,
                        "entry_price": price,
                        "entry_usdt": invest,
                        "entry_amount": amount,
                        "mature_price": mature_price,
                        "status": "Open",
                    }
                    ledger.open_note(note)
                    sim_capital -= invest
                    last_buy_tick[name] = tick
                    addlog(
                        f"[BUY] {name} tick {tick} price={price:.6f}",
                        verbose_int=2,
                        verbose_state=verbose,
                    )
    return sim_capital


def handle_sells(
    *,
    ledger: Ledger,
    name: str,
    tick: int,
    price: float,
    sim_capital: float,
    verbose: int,
) -> tuple[float, list]:
    """Close notes that have reached their target price.

    Returns updated capital and the list of notes closed at this tick.
    """
    to_close = []
    closed: list = []
    for note in ledger.get_active_notes():
        if note["window"] != name:
            continue
        if price >= note["mature_price"]:
            note["exit_tick"] = tick
            note["exit_price"] = price
            note["exit_usdt"] = note["entry_amount"] * price
            note["gain_usdt"] = note["exit_usdt"] - note["entry_usdt"]
            note["gain_pct"] = note["gain_usdt"] / note["entry_usdt"]
            note["status"] = "Closed"
            to_close.append(note)
    for note in to_close:
        ledger.close_note(note)
        sim_capital += note["exit_usdt"]
        closed.append(note)
    return sim_capital, closed
