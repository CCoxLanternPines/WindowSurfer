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


def summarize_simulation(*, ledger: RamLedger, start_capital: float, end_capital: float, total_ticks: int, verbose: int) -> None:
    """Log and persist simulation results."""

    addlog(
        f"[SIM] Completed {total_ticks} ticks. Realised PnL: {ledger.pnl:.2f}",
        verbose_int=1,
        verbose_state=verbose,
    )
    addlog(
        f"[SIM] Remaining capital: {end_capital:.2f}",
        verbose_int=1,
        verbose_state=verbose,
    )

    save_ledger(ledger, end_capital)
    summary = ledger.get_summary()
    addlog(
        f"[SIM] Ledger summary: {json.dumps(summary, indent=2)}",
        verbose_int=2,
        verbose_state=verbose,
    )
    addlog(
        f"[SIM] Starting capital: {start_capital:.2f}",
        verbose_int=2,
        verbose_state=verbose,
    )
    addlog(
        f"[SIM] Ending capital: {end_capital:.2f}",
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

                sim_capital = handle_sells(
                    ledger=ledger,
                    name=name,
                    tick=tick,
                    price=price,
                    sim_capital=sim_capital,
                    verbose=verbose,
                )

            pbar.update(1)

    summarize_simulation(
        ledger=ledger,
        start_capital=start_capital,
        end_capital=sim_capital,
        total_ticks=len(df),
        verbose=verbose,
    )

