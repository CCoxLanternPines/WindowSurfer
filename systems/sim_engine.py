from __future__ import annotations

"""Simple historical simulation engine using wave windows.

This module iterates over historical candle data and performs basic buy/sell
logic based purely on the position of the current price within a windowed wave
range. Configuration is loaded from ``settings/settings.json`` and all trades
are recorded in a lightweight in-memory ledger.
"""

from tqdm import tqdm

from systems.scripts.fetch_canles import fetch_candles
from systems.scripts.ledger import Ledger, save_ledger
from systems.scripts.handle_top_of_hour import handle_top_of_hour
from systems.utils.addlog import addlog
from systems.utils.settings_loader import load_settings
from systems.utils.resolve_symbol import resolve_ledger_settings
from systems.utils.path import find_project_root


def run_simulation(ledger_name: str, verbose: int = 0) -> None:
    """Run a historical simulation for ``ledger_name``."""
    settings = load_settings()
    ledger_config = resolve_ledger_settings(ledger_name, settings)
    tag = ledger_config["tag"].upper()

    root = find_project_root()
    sim_path = root / "data" / "tmp" / "simulation" / f"{tag}.json"
    if sim_path.exists():
        sim_path.unlink()

    windows = ledger_config.get("window_settings", {})
    if not windows:
        raise ValueError("No windows defined for ledger")

    sim_capital = float(settings.get("simulation_capital", 0))
    ledger = Ledger()

    addlog(
        f"[SIM] {ledger_name} | {tag} Starting simulation",
        verbose_int=1,
        verbose_state=verbose,
    )

    df = fetch_candles(tag)
    max_note_usdt = settings.get("general_settings", {}).get("max_note_usdt", sim_capital)
    min_note_usdt = settings.get("general_settings", {}).get("minimum_note_size", 0)

    last_buy_tick = {name: float("-inf") for name in windows}
    last_sell_tick = {name: float("-inf") for name in windows}
    buy_cooldown_skips = {name: 0 for name in windows}
    sell_cooldown_skips = {name: 0 for name in windows}
    min_roi_gate_hits = 0

    state = {
        "capital": sim_capital,
        "last_buy_tick": last_buy_tick,
        "last_sell_tick": last_sell_tick,
        "buy_cooldown_skips": buy_cooldown_skips,
        "sell_cooldown_skips": sell_cooldown_skips,
        "min_roi_gate_hits": min_roi_gate_hits,
    }

    total = len(df)
    with tqdm(total=total, desc="📉 Sim Progress", dynamic_ncols=True) as pbar:
        for tick in range(total):
            offset = total - tick - 1
            candle = df.iloc[tick].to_dict()

            handle_top_of_hour(
                tick=tick,
                candle=candle,
                ledger=ledger,
                ledger_name=ledger_name,
                settings=settings,
                sim=True,
                df=df,
                offset=offset,
                state=state,
                max_note_usdt=max_note_usdt,
                min_note_usdt=min_note_usdt,
                verbose=verbose,
            )

            pbar.update(1)

    addlog(
        f"[SIM] {ledger_name} | {tag} Completed {len(df)} ticks.",
        verbose_int=1,
        verbose_state=verbose,
    )

    final_tick = len(df) - 1 if total else -1
    final_price = float(df.iloc[-1]["close"])
    summary = ledger.get_account_summary(final_price)

    addlog(
        f"[DEBUG] {ledger_name} | {tag} Final tick: {final_tick}",
        verbose_int=3,
        verbose_state=verbose,
    )
    save_ledger(tag, ledger, sim=True, final_tick=final_tick, summary=summary)

    saved_summary = Ledger.load_ledger(tag, sim=True).get_account_summary(final_price)
    if (
        saved_summary["closed_notes"] != summary["closed_notes"]
        or saved_summary["realized_gain"] != summary["realized_gain"]
    ):
        addlog(
            f"[WARN] {ledger_name} | {tag} Summary/ledger mismatch: "
            f"closed_notes {summary['closed_notes']} vs {saved_summary['closed_notes']}, "
            f"realized_gain {summary['realized_gain']:.2f} vs {saved_summary['realized_gain']:.2f}",
            verbose_int=1,
            verbose_state=verbose,
        )

    addlog(
        f"[SIM] {ledger_name} | {tag} Final Price: ${summary['final_price']:.2f}",
        verbose_int=1,
        verbose_state=verbose,
    )
    addlog(
        f"[SIM] {ledger_name} | {tag} Total Coin Held: {summary['open_coin_amount']:.6f}",
        verbose_int=1,
        verbose_state=verbose,
    )
    addlog(
        f"[SIM] {ledger_name} | {tag} Final Value (USD): ${summary['total_value']:.2f}",
        verbose_int=1,
        verbose_state=verbose,
    )

    if verbose:
        addlog(
            f"[SIM] {ledger_name} | {tag} Buy cooldown skips: {state['buy_cooldown_skips']}",
            verbose_int=2,
            verbose_state=verbose,
        )
        addlog(
            f"[SIM] {ledger_name} | {tag} Sell cooldown skips: {state['sell_cooldown_skips']}",
            verbose_int=2,
            verbose_state=verbose,
        )
    addlog(
        f"[SIM] {ledger_name} | {tag} Min ROI gate hits: {state['min_roi_gate_hits']}",
        verbose_int=1,
        verbose_state=verbose,
    )

