from __future__ import annotations

"""Sell helper for the simulation engine."""

from typing import Dict, List, Tuple

from systems.scripts.ledger import Ledger
from systems.utils.trade_eval import evaluate_trade


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
    base_sell_cooldown: int,
    last_sell_tick: Dict[str, int],
) -> Tuple[float, List[Dict], int]:
    """Close notes that have reached their target price."""

    strategy_cfg = {
        "name": name,
        "cfg": cfg,
        "wave": wave,
        "sim_capital": sim_capital,
        "base_sell_cooldown": base_sell_cooldown,
    }

    updated_capital, closed, roi_skipped = evaluate_trade(
        "sell",
        price,
        ledger,
        strategy_cfg,
        last_sell_tick,
        tick,
        verbose,
    )
    return updated_capital, closed, roi_skipped
