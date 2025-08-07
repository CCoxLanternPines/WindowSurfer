from __future__ import annotations

"""Buy helper for the simulation engine."""

from typing import Dict, Tuple

from systems.scripts.ledger import Ledger
from systems.scripts.get_window_data import get_window_data
from systems.utils.addlog import addlog
from systems.utils.trade_eval import evaluate_trade


def evaluate_buy(
    *,
    ledger: Ledger,
    name: str,
    cfg: Dict,
    wave: Dict,
    tick: int,
    price: float,
    symbol: str | None,
    sim_capital: float,
    last_buy_tick: Dict[str, int],
    max_note_usdt: float,
    min_note_usdt: float,
    verbose: int,
) -> Tuple[float, bool]:
    """Attempt to open a new note if buy conditions are met."""

    if verbose >= 3:
        summary = get_window_data(wave=wave, price=price)
        tag = symbol or ledger.get_metadata().get("tag", "")
        addlog(
            (
                f"[DEBUG] {tag} | {name} | "
                f"Position: {summary['current_tunnel_position_avg']:.2f} | "
                f"Loudness: {summary['loudness_avg']:.2f} | "
                f"Slope: {summary['slope_direction_avg']:.2f} | "
                f"ROI est: {summary['highest_spike_avg']:.2f}"
            ),
            verbose_int=3,
            verbose_state=verbose,
        )

    strategy_cfg = {
        "name": name,
        "cfg": cfg,
        "wave": wave,
        "sim_capital": sim_capital,
        "max_note_usdt": max_note_usdt,
        "min_note_usdt": min_note_usdt,
    }

    updated_capital, skipped = evaluate_trade(
        "buy",
        price,
        ledger,
        strategy_cfg,
        last_buy_tick,
        tick,
        verbose,
    )
    return updated_capital, skipped
