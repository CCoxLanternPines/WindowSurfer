from __future__ import annotations

"""Buy signal evaluation used by simulation and live paths."""

from typing import Dict, Optional

from systems.scripts.ledger import Ledger
from systems.scripts.window_position_tools import get_trade_params
from systems.utils.addlog import addlog


def evaluate_buy(
    *,
    state: Dict,
    ledger: Ledger,
    strategy: str,
    cfg: Dict,
    wave: Dict,
    tick: int,
    price: float,
    cooldowns: Dict[str, Dict[str, int]],
    max_note_usdt: float,
    min_note_usdt: float,
    verbose: int,
) -> Optional[Dict]:
    """Return a buy signal for ``strategy`` if conditions are met."""

    trade = get_trade_params(
        current_price=price,
        window_high=wave["ceiling"],
        window_low=wave["floor"],
        config=cfg,
    )
    if verbose >= 3:
        addlog(
            f"[DEBUG][BUY] Window={strategy} price={price:.6f} pos_pct={trade['pos_pct']:.2f} "
            f"ceiling={wave['ceiling']:.6f} floor={wave['floor']:.6f} "
            f"buy_mult={trade['buy_multiplier']:.2f} "
            f"buy_cd_mult={trade['buy_cooldown_multiplier']:.2f}",
            verbose_int=3,
            verbose_state=verbose,
        )
    if trade["in_dead_zone"]:
        return None

    cd = cooldowns.get(strategy, {}).get("buy", 0)
    if cd > 0:
        return None

    open_for_window = [
        n
        for n in ledger.get_open_notes()
        if n.get("strategy") == strategy and n.get("status") == "open"
    ]
    if len(open_for_window) >= cfg.get("max_open_notes", 0):
        return None

    capital = state.get("capital", 0.0)
    base_invest = capital * cfg.get("investment_fraction", 0)
    adjusted_invest = base_invest * trade["buy_multiplier"]
    invest = min(adjusted_invest, max_note_usdt)
    if invest < min_note_usdt or invest > capital or invest <= 0:
        return None

    return {
        "strategy": strategy,
        "amount_usd": invest,
        "price": price,
    }
