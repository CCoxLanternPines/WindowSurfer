from __future__ import annotations

"""Buy helpers used by both live and simulation paths."""

from typing import Dict, List, Tuple

from systems.scripts.ledger import Ledger
from systems.scripts.get_window_data import get_window_data
from systems.scripts.window_position_tools import get_trade_params
from systems.utils.addlog import addlog


def compute_buy_signals(
    *,
    price: float,
    wave: Dict,
    cfg: Dict,
    ledger: Ledger,
    tick: int,
    name: str,
    sim_capital: float,
    last_buy_tick: Dict[str, int],
    max_note_usdt: float,
    min_note_usdt: float,
    verbose: int,
) -> Tuple[List[Dict], bool]:
    """Return potential buy notes without executing them.

    Returns a tuple of ``(signals, skipped)`` where ``signals`` is a list of
    note dictionaries to be opened and ``skipped`` indicates a cooldown skip.
    """

    trade_params = get_trade_params(
        current_price=price,
        window_high=wave["ceiling"],
        window_low=wave["floor"],
        config=cfg,
    )

    if trade_params["in_dead_zone"]:
        return [], False

    base_cd = cfg.get("buy_cooldown", 0)
    adjusted_cd = int(base_cd / trade_params["buy_cooldown_multiplier"])
    if tick - last_buy_tick.get(name, float("-inf")) < adjusted_cd:
        return [], True

    open_for_window = [n for n in ledger.get_active_notes() if n["window"] == name]
    if len(open_for_window) >= cfg.get("max_open_notes", 0):
        return [], False

    base_note_count = sim_capital * cfg.get("investment_fraction", 0)
    adjusted_note_count = base_note_count * trade_params["buy_multiplier"]
    invest = min(adjusted_note_count, max_note_usdt)
    if invest < min_note_usdt or invest > sim_capital:
        return [], False

    amount = invest / price
    metadata = ledger.get_metadata()
    note = {
        "asset": metadata.get("asset"),
        "tag": metadata.get("tag"),
        "window": name,
        "entry_tick": tick,
        "buy_tick": tick,
        "entry_price": price,
        "entry_amount": amount,
        "status": "Open",
    }
    return [{"note": note, "invest": invest}], False


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
    """Execute buy signals and update capital."""

    if verbose >= 3:
        summary = get_window_data(wave=wave, price=price)
        metadata = ledger.get_metadata()
        asset = metadata.get("asset", "")
        tag = symbol or metadata.get("tag", "")
        addlog(
            (
                f"[DEBUG] {asset} ({tag}) | {name} | "
                f"Position: {summary['current_tunnel_position_avg']:.2f} | "
                f"Loudness: {summary['loudness_avg']:.2f} | "
                f"Slope: {summary['slope_direction_avg']:.2f} | "
                f"ROI est: {summary['highest_spike_avg']:.2f}"
            ),
            verbose_int=3,
            verbose_state=verbose,
        )

    signals, skipped = compute_buy_signals(
        price=price,
        wave=wave,
        cfg=cfg,
        ledger=ledger,
        tick=tick,
        name=name,
        sim_capital=sim_capital,
        last_buy_tick=last_buy_tick,
        max_note_usdt=max_note_usdt,
        min_note_usdt=min_note_usdt,
        verbose=verbose,
    )

    for sig in signals:
        ledger.open_note(sig["note"])
        sim_capital -= sig["invest"]
        last_buy_tick[name] = tick
    return sim_capital, skipped
