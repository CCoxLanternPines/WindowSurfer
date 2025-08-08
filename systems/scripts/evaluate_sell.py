from __future__ import annotations

"""Sell helpers used by both live and simulation paths."""

from typing import Dict, List, Tuple

from systems.scripts.ledger import Ledger
from systems.scripts.window_position_tools import get_trade_params
from systems.utils.addlog import addlog


def compute_sell_signals(
    *,
    price: float,
    wave: Dict,
    cfg: Dict,
    ledger: Ledger,
    tick: int,
    name: str,
    base_sell_cooldown: int,
    last_sell_tick: Dict[str, int],
    verbose: int,
) -> Tuple[List[Dict], int]:
    """Return notes to close without executing them.

    Returns a tuple of ``(signals, roi_skipped)`` where ``signals`` contains
    note dictionaries annotated with exit details.
    """

    trade = get_trade_params(
        price, wave["ceiling"], wave["floor"], cfg
    )
    if trade["in_dead_zone"]:
        return [], 0

    notes = [n for n in ledger.get_active_notes() if n["window"] == name]
    notes.sort(
        key=lambda n: (price - n["entry_price"]) / n["entry_price"],
        reverse=True,
    )

    signals: List[Dict] = []
    roi_skipped = 0
    for note in notes:
        gain_pct = (price - note["entry_price"]) / note["entry_price"]
        trade_note = get_trade_params(
            price,
            wave["ceiling"],
            wave["floor"],
            cfg,
            entry_price=note["entry_price"],
        )
        maturity_roi = trade_note["maturity_roi"]
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
        if tick - last_sell_tick.get(name, float("-inf")) < adjusted_cd:
            continue

        gain = (price - note["entry_price"]) * note["entry_amount"]
        base = note["entry_price"] * note["entry_amount"] or 1
        signal = note.copy()
        signal.update(
            {
                "exit_tick": tick,
                "exit_price": price,
                "exit_ts": tick,
                "gain": gain,
                "gain_pct": gain / base,
                "status": "Closed",
            }
        )
        signals.append(signal)

    return signals, roi_skipped


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

    metadata = ledger.get_metadata()
    asset = metadata.get("asset", "")
    tag = metadata.get("tag", "")

    if verbose >= 3:
        addlog(
            f"[DEBUG][SELL] {asset} ({tag}) | {name}",
            verbose_int=3,
            verbose_state=verbose,
        )

    signals, roi_skipped = compute_sell_signals(
        price=price,
        wave=wave,
        cfg=cfg,
        ledger=ledger,
        tick=tick,
        name=name,
        base_sell_cooldown=base_sell_cooldown,
        last_sell_tick=last_sell_tick,
        verbose=verbose,
    )

    closed: List[Dict] = []
    if signals:
        last_sell_tick[name] = tick
    for sig in signals:
        ledger.close_note(sig)
        sim_capital += sig["entry_amount"] * price
        closed.append(sig)
    return sim_capital, closed, roi_skipped
