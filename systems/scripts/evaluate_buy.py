from __future__ import annotations

"""Buy helper for the simulation engine."""

from typing import Dict, Tuple

from systems.scripts.ledger import Ledger
from systems.utils.logger import addlog


def evaluate_buy(
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
) -> Tuple[float, bool]:
    """Attempt to open a new note if buy conditions are met.

    Returns updated capital and whether the attempt was skipped due to cooldown.
    """
    position = wave["position_in_window"]
    buy_cooldown = cfg.get("buy_cooldown", 0)
    if tick - last_buy_tick.get(name, float("-inf")) < buy_cooldown:
        return sim_capital, True
    if position <= cfg.get("buy_floor", 0):
        open_for_window = [n for n in ledger.get_active_notes() if n["window"] == name]
        if len(open_for_window) < cfg.get("max_open_notes", 0):
            invest = sim_capital * cfg.get("investment_fraction", 0)
            invest = min(invest, max_note_usdt)
            if invest >= min_note_usdt and invest <= sim_capital:
                amount = invest / price
                configured_mature = wave["floor"] + wave["range"] * cfg.get("sell_ceiling", 1)
                configured_roi = (configured_mature - price) / price
                min_roi = cfg.get("min_roi_pct", 0) / 100.0
                target_roi = max(configured_roi, min_roi)
                mature_price = price * (1 + target_roi)
                note = {
                    "window": name,
                    "entry_tick": tick,
                    "buy_tick": tick,
                    "entry_price": price,
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
    return sim_capital, False
