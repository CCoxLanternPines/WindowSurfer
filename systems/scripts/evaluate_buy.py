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

    pre_count = len(ledger.get_open_notes())
    updated_capital, skipped = evaluate_trade(
        "buy",
        price,
        ledger,
        strategy_cfg,
        last_buy_tick,
        tick,
        verbose,
    )

    # ------------------------------------------------------------------
    # Optional baked targets
    # ------------------------------------------------------------------
    if not skipped and cfg.get("use_baked_targets"):
        post_notes = ledger.get_open_notes()
        if len(post_notes) > pre_count:
            note = post_notes[-1]
            entry_price = note.get("entry_price", 0.0)
            floor = wave.get("floor", 0.0)
            ceiling = wave.get("ceiling", 0.0)
            window_range = ceiling - floor
            if window_range != 0:
                pos_pct = ((entry_price - floor) / window_range) * 2 - 1
                mirrored_pos = -pos_pct
                target_price = floor + ((mirrored_pos + 1) / 2) * window_range
                maturity_mult = cfg.get("maturity_multiplier", 1.0)
                structural_target = entry_price + (
                    (target_price - entry_price) * maturity_mult
                )
            else:
                structural_target = entry_price
            required_min_roi = cfg.get("required_min_roi", 0.0)
            maturity_price = max(
                entry_price * (1 + required_min_roi), structural_target
            )
            note["maturity_price"] = maturity_price

    return updated_capital, skipped
