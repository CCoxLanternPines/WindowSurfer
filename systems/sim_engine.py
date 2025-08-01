from __future__ import annotations

"""Simple historical simulation engine using wave windows.

This module iterates over historical candle data and performs basic buy/sell
logic based purely on the position of the current price within a windowed wave
range. Configuration is loaded from ``settings/settings.json`` and all trades
are recorded in a lightweight in-memory ledger.
"""

from tqdm import tqdm

from systems.scripts.fetch_canles import fetch_candles
from systems.scripts.ledger import Ledger
from systems.scripts.evaluate_buy import evaluate_buy
from systems.scripts.evaluate_sell import evaluate_sell
from systems.scripts.get_window_data import get_wave_window_data_df
from systems.utils.logger import addlog
from systems.utils.settings_loader import load_settings


def run_simulation(tag: str, verbose: int = 0) -> None:
    """Run a historical simulation for ``tag``."""
    settings = load_settings()
    tag = tag.upper()
    symbol_meta = settings.get("symbol_settings", {}).get(tag)
    if symbol_meta is None:
        raise ValueError(f"Unknown symbol tag: {tag}")

    windows = settings.get("general_settings", {}).get("windows", {})
    if not windows:
        raise ValueError("No windows defined in settings['general_settings']['windows']")

    sim_capital = float(settings.get("simulation_capital", 0))
    ledger = Ledger()

    addlog(f"[SIM] Starting simulation for {tag}", verbose_int=1, verbose_state=verbose)

    df = fetch_candles(tag)
    max_note_usdt = settings.get("general_settings", {}).get("max_note_usdt", sim_capital)
    min_note_usdt = settings.get("general_settings", {}).get("minimum_note_size", 0)

    last_buy_tick = {name: float("-inf") for name in windows}
    active_cooldowns = {name: 0.0 for name in windows}

    total = len(df)
    with tqdm(total=total, desc="📉 Sim Progress", dynamic_ncols=True) as pbar:
        for tick in range(total):
            offset = total - tick - 1
            price = float(df.iloc[tick]["close"])

            for name, cfg in windows.items():
                if active_cooldowns[name] > 0:
                    active_cooldowns[name] = max(0.0, active_cooldowns[name] - 1)
                wave = get_wave_window_data_df(
                    df,
                    window=cfg["window_size"],
                    candle_offset=offset,
                )
                if not wave:
                    continue

                sim_capital = evaluate_buy(
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
                cooldown = cfg.get("cooldown", 0)

                # Check if a new note was opened for this window
                if any(n["entry_tick"] == tick and n["window"] == name for n in ledger.get_active_notes()):
                    delta_24h = abs(wave.get("trend_direction_delta_window", 0.0))
                    active_cooldowns[name] = max(0.0, cooldown - delta_24h)


                if verbose >= 3 and wave:
                    trend_direction_delta_window = wave.get("trend_direction_delta_window")
                    position_in_window = wave.get("position_in_window")
                    trend_direction_delta_window = f"{trend_direction_delta_window:+.4f}" if isinstance(trend_direction_delta_window, (float, int)) else "N/A"

                    addlog(
                        f"[DEBUG] Tick {tick} | Window: {name} | trend_direction_delta_window: {trend_direction_delta_window} | position_in_window {position_in_window} | {active_cooldowns[name]:.2f}",
                        verbose_int=3,
                        verbose_state=verbose,
                    )
                if active_cooldowns[name] <= 0:
                    sim_capital, closed = evaluate_sell(
                        ledger=ledger,
                        name=name,
                        tick=tick,
                        price=price,
                        sim_capital=sim_capital,
                        verbose=verbose,
                    )
                    if closed:
                        active_cooldowns[name] = max(0.0, cooldown)
                        for note in closed:
                            addlog(
                                (
                                    f"[SELL] Tick {tick} | Window: {note['window']} | "
                                    f"Gain: +${note['gain']:.2f} ({note['gain_pct']:.2%})"
                                ),
                                verbose_int=2,
                                verbose_state=verbose,
                            )
            pbar.update(1)

    print(f"[SIM] Completed {len(df)} ticks.")

    final_price = float(df.iloc[-1]["close"])
    summary = ledger.get_account_summary(final_price)

    Ledger.save_ledger(tag, ledger)

    print(f"Final Price: ${summary['final_price']:.2f}")
    print(f"Total Coin Held: {summary['open_coin_amount']:.6f}")
    print(f"Final Value (USD): ${summary['total_value']:.2f}")

