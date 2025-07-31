from __future__ import annotations

"""Evaluate sell conditions with optional progressive scaling."""

from typing import Dict, List, Tuple

from scripts.ledger_manager import RamLedger
from systems.utils.logger import addlog


def evaluate_sell(
    *,
    ledger: RamLedger,
    name: str,
    cfg: Dict,
    wave: Dict,
    tick: int,
    total_ticks: int,
    price: float,
    sim_capital: float,
    last_sell_tick: Dict[str, int],
    verbose: int,
) -> Tuple[float, List[Dict]]:
    """Close notes that meet sell conditions.

    Returns updated capital and list of notes closed at this tick.
    """

    cooldown = cfg.get("cooldown", 0)
    if tick - last_sell_tick.get(name, -9999) < cooldown:
        return sim_capital, []

    position = wave["position_in_window"]
    base_ceiling = cfg.get("sell_ceiling", 1.0)

    if cfg.get("sell_scale"):
        progress = tick / total_ticks if total_ticks else 1.0
        progress = min(progress, 1.0)
        threshold = base_ceiling * progress
    else:
        threshold = base_ceiling

    closed: List[Dict] = []
    if position >= threshold:
        active = [n for n in ledger.get_active_notes() if n["window"] == name]
        if active:
            active.sort(key=lambda n: n.get("entry_usdt", 0), reverse=True)
            note = active[0]
            note["exit_tick"] = tick
            note["exit_price"] = price
            note["exit_usdt"] = note["entry_amount"] * price
            note["gain_usdt"] = note["exit_usdt"] - note["entry_usdt"]
            note["gain_pct"] = note["gain_usdt"] / note["entry_usdt"]
            note["status"] = "Closed"
            ledger.close_note(note)
            sim_capital += note["exit_usdt"]
            last_sell_tick[name] = tick
            closed.append(note)
            addlog(
                f"[SELL] {name} tick {tick} price={price:.6f}",
                verbose_int=2,
                verbose_state=verbose,
            )
    return sim_capital, closed
