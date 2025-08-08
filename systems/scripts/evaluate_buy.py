from __future__ import annotations

"""Buy helper for simulation and live signal computation."""

from dataclasses import dataclass
from typing import Dict, List, Tuple

from systems.scripts.ledger import Ledger
from systems.scripts.get_window_data import get_window_data
from systems.scripts.window_position_tools import get_trade_params
from systems.utils.addlog import addlog


@dataclass
class BuySignal:
    """Container representing a buy decision."""

    name: str
    price: float
    amount: float
    invest_usdt: float


def compute_buy_signals(
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
) -> Tuple[List[BuySignal], bool]:
    """Return buy signals for the given window and price."""

    trade = get_trade_params(
        current_price=price,
        window_high=wave["ceiling"],
        window_low=wave["floor"],
        config=cfg,
    )
    if trade["in_dead_zone"]:
        return [], False

    base_cd = cfg.get("buy_cooldown", 0)
    adjusted_cd = int(base_cd / trade["buy_cooldown_multiplier"])
    if tick - last_buy_tick.get(name, float("-inf")) < adjusted_cd:
        return [], True

    open_for_window = [n for n in ledger.get_active_notes() if n.get("window") == name]
    if len(open_for_window) >= cfg.get("max_open_notes", 0):
        return [], False

    base_invest = sim_capital * cfg.get("investment_fraction", 0)
    adjusted_invest = base_invest * trade["buy_multiplier"]
    invest = min(adjusted_invest, max_note_usdt)
    if invest < min_note_usdt or invest > sim_capital:
        return [], False

    amount = invest / price if price else 0.0
    signal = BuySignal(name=name, price=price, amount=amount, invest_usdt=invest)
    return [signal], False


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

    signals, skipped = compute_buy_signals(
        ledger=ledger,
        name=name,
        cfg=cfg,
        wave=wave,
        tick=tick,
        price=price,
        sim_capital=sim_capital,
        last_buy_tick=last_buy_tick,
        max_note_usdt=max_note_usdt,
        min_note_usdt=min_note_usdt,
    )

    for signal in signals:
        note = {
            "window": signal.name,
            "entry_tick": tick,
            "buy_tick": tick,
            "entry_price": signal.price,
            "entry_amount": signal.amount,
            "status": "Open",
        }
        ledger.open_note(note)
        sim_capital -= signal.invest_usdt
        last_buy_tick[name] = tick
        addlog(
            f"[BUY] {name} tick {tick} price={price:.6f}",
            verbose_int=2,
            verbose_state=verbose,
        )

    return sim_capital, skipped
