from __future__ import annotations

"""Evaluate buy conditions with optional progressive scaling."""

from typing import Dict

from scripts.ledger_manager import RamLedger
from systems.utils.logger import addlog


def evaluate_buy(
    *,
    ledger: RamLedger,
    name: str,
    cfg: Dict,
    wave: Dict,
    tick: int,
    total_ticks: int,
    price: float,
    sim_capital: float,
    last_buy_tick: Dict[str, int],
    max_note_usdt: float,
    min_note_usdt: float,
    verbose: int,
) -> float:
    """Attempt to open a new note if buy conditions are met."""

    position = wave["position_in_window"]
    base_floor = cfg.get("buy_floor", 0.0)

    if cfg.get("buy_scale"):
        progress = tick / total_ticks if total_ticks else 1.0
        progress = min(progress, 1.0)
        threshold = base_floor * (1.0 - progress)
    else:
        threshold = base_floor

    if position <= threshold:
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
