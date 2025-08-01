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
from systems.scripts.trade_logic import maybe_buy
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
    starting_capital = sim_capital
    ledger = Ledger(sim_capital)

    addlog(f"[SIM] Starting simulation for {tag}", verbose_int=1, verbose_state=verbose)

    df = fetch_candles(tag)
    max_note_usdt = settings.get("general_settings", {}).get("max_note_usdt", sim_capital)
    min_note_usdt = settings.get("general_settings", {}).get("minimum_note_size", 0)

    last_buy_tick = {name: float("-inf") for name in windows}
    last_sell_tick: dict[str, int] = {}
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
                    active_notes = [
                        n for n in ledger.get_active_notes() if n["window"] == name
                    ]
                    if active_notes:
                        active_notes.sort(
                            key=lambda n: n.get("entry_usdt", 0), reverse=True
                        )
                        note = active_notes[0]
                        if price >= note["mature_price"]:
                            note["exit_tick"] = tick
                            note["exit_price"] = price
                            note["exit_usdt"] = note["entry_amount"] * price
                            note["gain_usdt"] = note["exit_usdt"] - note["entry_usdt"]
                            note["gain_pct"] = note["gain_usdt"] / note["entry_usdt"]
                            note["status"] = "Closed"
                            ledger.close_note(note)
                            sim_capital += note["exit_usdt"]
                            last_sell_tick[name] = tick
                            active_cooldowns[name] = max(0.0, cooldown)
                            addlog(
                                (
                                    f"[SELL] Tick {tick} | Window: {note['window']} | "
                                    f"Gain: +${note['gain_usdt']:.2f} ({note['gain_pct']:.2%})"
                                ),
                                verbose_int=2,
                                verbose_state=verbose,
                            )
            ledger.set_capital(sim_capital)
            pbar.update(1)

    Ledger.save_ledger(tag, ledger)
    summary = ledger.get_account_summary(starting_capital)
    print(f"[SIM] Completed {len(df)} ticks.")
    for k, v in summary.items():
        label = k.replace("_", " ").title()
        print(f"[SIM] {label}: {v}")

