from __future__ import annotations

"""Buy helper for the simulation engine."""

from typing import Dict, Tuple

from systems.scripts.ledger import Ledger
from systems.utils.addlog import addlog
from systems.scripts.window_position_tools import get_trade_params


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
    trade_params = get_trade_params(
        current_price=price,
        window_high=wave["ceiling"],
        window_low=wave["floor"],
        config=cfg,
    )
    if trade_params["in_dead_zone"]:
        return sim_capital, False

    base_buy_cooldown = cfg.get("buy_cooldown", 0)
    adjusted_cooldown_ticks = int(
        base_buy_cooldown / trade_params["buy_cooldown_multiplier"]
    )
    if tick - last_buy_tick.get(name, float("-inf")) < adjusted_cooldown_ticks:
        return sim_capital, True

    open_for_window = [n for n in ledger.get_active_notes() if n["window"] == name]
    if len(open_for_window) < cfg.get("max_open_notes", 0):
        base_note_count = sim_capital * cfg.get("investment_fraction", 0)
        adjusted_note_count = base_note_count * trade_params["buy_multiplier"]
        invest = min(adjusted_note_count, max_note_usdt)
        if invest >= min_note_usdt and invest <= sim_capital:
            amount = invest / price
            note = {
                "window": name,
                "entry_tick": tick,
                "buy_tick": tick,
                "entry_price": price,
                "entry_amount": amount,
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
