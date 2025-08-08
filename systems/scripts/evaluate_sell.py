from __future__ import annotations

"""Sell helper for simulation and live signal computation."""

from dataclasses import dataclass
from typing import Dict, List, Tuple

from systems.scripts.ledger import Ledger
from systems.scripts.window_position_tools import get_trade_params


@dataclass
class SellSignal:
    """Container representing a sell decision."""

    note: Dict
    price: float
    gain: float
    gain_pct: float


def compute_sell_signals(
    *,
    ledger: Ledger,
    name: str,
    tick: int,
    price: float,
    wave: Dict,
    cfg: Dict,
    base_sell_cooldown: int,
    last_sell_tick: Dict[str, int],
) -> Tuple[List[SellSignal], int]:
    """Return sell signals for the given window and price."""

    trade = get_trade_params(
        current_price=price,
        window_high=wave["ceiling"],
        window_low=wave["floor"],
        config=cfg,
    )
    if trade["in_dead_zone"]:
        return [], 0

    notes = [n for n in ledger.get_active_notes() if n.get("window") == name]
    notes.sort(
        key=lambda n: (price - n["entry_price"]) / n["entry_price"],
        reverse=True,
    )

    signals: List[SellSignal] = []
    roi_skipped = 0
    for note in notes:
        gain_pct = (price - note["entry_price"]) / note["entry_price"]
        trade_note = get_trade_params(
            current_price=price,
            window_high=wave["ceiling"],
            window_low=wave["floor"],
            config=cfg,
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
        signals.append(
            SellSignal(note=note, price=price, gain=gain, gain_pct=gain_pct)
        )

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

    signals, roi_skipped = compute_sell_signals(
        ledger=ledger,
        name=name,
        tick=tick,
        price=price,
        wave=wave,
        cfg=cfg,
        base_sell_cooldown=base_sell_cooldown,
        last_sell_tick=last_sell_tick,
    )

    closed: List[Dict] = []
    for signal in signals:
        note = signal.note
        note["exit_tick"] = tick
        note["exit_price"] = signal.price
        note["exit_ts"] = tick
        note["gain"] = signal.gain
        note["gain_pct"] = signal.gain_pct
        note["status"] = "Closed"
        ledger.close_note(note)
        sim_capital += note["entry_amount"] * signal.price
        closed.append(note)
        last_sell_tick[name] = tick

    return sim_capital, closed, roi_skipped
