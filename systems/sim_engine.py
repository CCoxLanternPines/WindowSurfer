from __future__ import annotations

"""Simple historical simulation engine using wave windows.

This module iterates over historical candle data and performs basic buy/sell
logic based purely on the position of the current price within a windowed wave
range. Configuration is loaded from ``settings/settings.json`` and all trades
are recorded in a lightweight in-memory ledger.
"""

import json

from tqdm import tqdm

from scripts.fetch_canles import fetch_candles
from scripts.ledger_manager import RamLedger, save_ledger
from scripts.trade_logic import handle_sells, maybe_buy
from systems.scripts.get_window_data import get_wave_window_data_df
from systems.utils.logger import addlog
from systems.utils.settings_loader import load_settings


# ---------------------------------------------------------------------------
# Core simulation logic
# ---------------------------------------------------------------------------


def summarize_simulation(
    *,
    ledger: RamLedger,
    start_capital: float,
    idle_capital: float,
    realised_pnl: float,
    open_value: float,
    end_value: float,
    total_ticks: int,
    verbose: int,
) -> None:
    """Log and persist simulation results."""

    addlog(f"[SIM] Completed {total_ticks} ticks.", verbose_int=1, verbose_state=verbose)
    addlog(
        f"[SIM] Starting capital: {start_capital:.2f}",
        verbose_int=1,
        verbose_state=verbose,
    )
    addlog(
        f"[SIM] Realised PnL: {realised_pnl:.2f}",
        verbose_int=1,
        verbose_state=verbose,
    )
    addlog(
        f"[SIM] Idle capital: {idle_capital:.2f}",
        verbose_int=1,
        verbose_state=verbose,
    )
    addlog(
        f"[SIM] Open note value: {open_value:.2f}",
        verbose_int=1,
        verbose_state=verbose,
    )
    addlog(
        f"[SIM] Ending value: {end_value:.2f}",
        verbose_int=1,
        verbose_state=verbose,
    )

    save_ledger(ledger, end_value)
    summary = ledger.get_summary()
    addlog(
        f"[SIM] Ledger summary: {json.dumps(summary, indent=2)}",
        verbose_int=2,
        verbose_state=verbose,
    )


def run_simulation(tag: str, window: str, verbose: int = 0) -> None:  # noqa: ARG001
    """Run a historical simulation for ``tag``."""
    settings = load_settings()
    tag = tag.upper()
    symbol_meta = settings.get("symbol_settings", {}).get(tag)
    if symbol_meta is None:
        raise ValueError(f"Unknown symbol tag: {tag}")

    sim_capital = float(settings.get("simulation_capital", 0))
    start_capital = sim_capital
    realised_pnl = 0.0
    ledger = RamLedger()

    addlog(f"[SIM] Starting simulation for {tag}", verbose_int=1, verbose_state=verbose)

    df = fetch_candles(tag)
    windows = settings.get("general_settings", {}).get("windows", {})
    max_note_usdt = settings.get("general_settings", {}).get("max_note_usdt", sim_capital)
    min_note_usdt = settings.get("general_settings", {}).get("minimum_note_size", 0)

    last_buy_tick = {name: float("-inf") for name in windows}

    with tqdm(total=len(df), desc="📉 Sim Progress", dynamic_ncols=True) as pbar:
        for tick in range(len(df)):
            current_df = df.iloc[: tick + 1]
            price = float(current_df.iloc[-1]["close"])

            for name, cfg in windows.items():
                wave = get_wave_window_data_df(current_df, cfg["window_size"])
                if not wave:
                    continue

                sim_capital = maybe_buy(
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
                    verbose=verbose,
                )
                before_pnl = ledger.pnl
                sim_capital = handle_sells(
                    ledger=ledger,
                    name=name,
                    tick=tick,
                    price=price,
                    sim_capital=sim_capital,
                    verbose=verbose,
                )
                realised_gain = ledger.pnl - before_pnl
                if realised_gain:
                    realised_pnl += realised_gain
                    sim_capital -= realised_gain

            pbar.update(1)

    last_price = float(df.iloc[-1]["close"])
    open_value = sum(
        n["entry_amount"] * last_price for n in ledger.get_active_notes()
    )
    end_value = sim_capital + open_value + realised_pnl
    summarize_simulation(
        ledger=ledger,
        start_capital=start_capital,
        idle_capital=sim_capital,
        realised_pnl=realised_pnl,
        open_value=open_value,
        end_value=end_value,
        total_ticks=len(df),
        verbose=verbose,
    )

