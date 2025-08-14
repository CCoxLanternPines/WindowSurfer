from __future__ import annotations

"""Unified trade evaluation helper for buy and sell paths."""

from typing import Dict, List, Tuple

from systems.scripts.window_utils import get_trade_params
from systems.utils.addlog import addlog
from systems.scripts.trade_apply import apply_buy, apply_sell


def evaluate_trade(
    trade_type: str,
    current_price: float,
    ledger,
    strategy_cfg: Dict,
    cooldown_tracker: Dict[str, int],
    tick: int,
    verbose: int,
):
    """Evaluate and execute a trade of ``trade_type``.

    Parameters
    ----------
    trade_type: ``"buy"`` or ``"sell"``
    current_price: float
    ledger: Ledger instance
    strategy_cfg: Dict containing window/config information
    cooldown_tracker: mapping of window -> last tick executed
    tick: current tick
    verbose: verbosity level
    """

    name = strategy_cfg["name"]
    cfg = strategy_cfg["cfg"]
    wave = strategy_cfg["wave"]
    sim_capital = strategy_cfg["sim_capital"]

    if trade_type == "buy":
        max_note_usdt = strategy_cfg["max_note_usdt"]
        min_note_usdt = strategy_cfg["min_note_usdt"]

        trade_params = get_trade_params(
            current_price=current_price,
            window_high=wave["ceiling"],
            window_low=wave["floor"],
            config=cfg,
        )
        if verbose >= 3:
            addlog(
                f"[DEBUG][BUY] Window={name} price={current_price:.6f} pos_pct={trade_params['pos_pct']:.2f} "
                f"ceiling={wave['ceiling']:.6f} floor={wave['floor']:.6f} "
                f"buy_mult={trade_params['buy_multiplier']:.2f} "
                f"buy_cd_mult={trade_params['buy_cooldown_multiplier']:.2f}",
                verbose_int=3,
                verbose_state=verbose,
            )

        if trade_params["in_dead_zone"]:
            return sim_capital, False

        base_cd = cfg.get("buy_cooldown", 0)
        adjusted_cd = int(base_cd / trade_params["buy_cooldown_multiplier"])
        if tick - cooldown_tracker.get(name, float("-inf")) < adjusted_cd:
            return sim_capital, True

        base_note_count = sim_capital * cfg.get("investment_fraction", 0)
        adjusted_note_count = base_note_count * trade_params["buy_multiplier"]
        invest = min(adjusted_note_count, max_note_usdt)
        if invest >= min_note_usdt and invest <= sim_capital:
            amount = invest / current_price
            result = {
                "filled_amount": amount,
                "avg_price": current_price,
                "timestamp": tick,
            }
            meta = {
                "window_name": name,
                "window_size": cfg.get("window_size"),
            }
            state = {"capital": sim_capital}
            apply_buy(
                ledger=ledger,
                window_name=name,
                t=tick,
                meta=meta,
                result=result,
                state=state,
            )
            sim_capital = state.get("capital", sim_capital)
            cooldown_tracker[name] = tick
            addlog(
                f"[BUY] {name} tick {tick} price={current_price:.6f}",
                verbose_int=2,
                verbose_state=verbose,
            )
        return sim_capital, False

    if trade_type == "sell":
        base_sell_cooldown = strategy_cfg["base_sell_cooldown"]
        trade = get_trade_params(current_price, wave["ceiling"], wave["floor"], cfg)
        if trade["in_dead_zone"]:
            return sim_capital, [], 0

        to_close: List[Dict] = []
        roi_skipped = 0
        notes = [n for n in ledger.get_active_notes() if n["window"] == name]
        notes.sort(
            key=lambda n: (current_price - n["entry_price"]) / n["entry_price"],
            reverse=True,
        )
        for note in notes:
            gain_pct = (current_price - note["entry_price"]) / note["entry_price"]
            trade_note = get_trade_params(
                current_price,
                wave["ceiling"],
                wave["floor"],
                cfg,
                entry_price=note["entry_price"],
            )
            maturity_roi = trade_note["maturity_roi"]
            if maturity_roi is not None:
                addlog(
                    f"[DEBUG][SELL] gain_pct={gain_pct:.2%} maturity_roi={maturity_roi:.2%}",
                    verbose_int=3,
                    verbose_state=verbose,
                )
            else:
                addlog(
                    f"[DEBUG][SELL] gain_pct={gain_pct:.2%} maturity_roi=None",
                    verbose_int=3,
                    verbose_state=verbose,
                )
            if (
                maturity_roi is None
                or maturity_roi <= 0
                or gain_pct < 0
                or gain_pct < maturity_roi
            ):
                roi_skipped += 1
                continue
            adjusted_cd = int(
                base_sell_cooldown / trade_note["sell_cooldown_multiplier"]
            )
            if tick - cooldown_tracker.get(name, float("-inf")) < adjusted_cd:
                continue
            gain = (current_price - note["entry_price"]) * note["entry_amount"]
            note["exit_tick"] = tick
            note["exit_price"] = current_price
            note["exit_ts"] = tick
            note["gain"] = gain
            base = note["entry_price"] * note["entry_amount"] or 1
            note["gain_pct"] = gain / base
            note["status"] = "Closed"
            to_close.append(note)
            cooldown_tracker[name] = tick

        closed: List[Dict] = []
        for note in to_close:
            res = {
                "filled_amount": note.get("entry_amount", 0.0),
                "avg_price": current_price,
                "timestamp": tick,
            }
            state = {"capital": sim_capital}
            apply_sell(ledger=ledger, note=note, t=tick, result=res, state=state)
            sim_capital = state.get("capital", sim_capital)
            closed.append(note)
        return sim_capital, closed, roi_skipped

    raise ValueError("trade_type must be 'buy' or 'sell'")

