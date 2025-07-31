from __future__ import annotations

"""Simple historical simulation engine using wave windows.

This module iterates over historical candle data and performs basic buy/sell
logic based purely on the position of the current price within a windowed wave
range. Configuration is loaded from ``settings/settings.json`` and all trades
are recorded in a lightweight in-memory ledger.
"""

import json

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm is missing
    class tqdm:  # type: ignore
        def __init__(self, total=None, **kwargs):
            self.total = total

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: D401 - silence pydocstyle
            return False

        def update(self, *args, **kwargs):
            pass

from scripts.fetch_canles import fetch_candles
from scripts.ledger_manager import RamLedger, save_ledger
from scripts.evaluate_buy import evaluate_buy
from scripts.evaluate_sell import evaluate_sell
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
    total_ticks: int,
    verbose: int,
) -> None:
    """Log and persist simulation results.

    The final account value is derived from ``idle_capital`` and the
    mark-to-market value of any open notes. ``realised_pnl`` is reported for
    informational purposes only and is not added to the ending value since it
    has already been credited to ``idle_capital`` when notes were closed.
    """

    end_value = idle_capital + open_value

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


def run_simulation(tag: str, *, settings: dict | None = None, verbose: int = 0) -> None:
    """Run a historical simulation for ``tag``."""
    print("[SIM] Simulation engine entered")
    if settings is None:
        settings = load_settings()
    print("[SIM] Settings loaded:", list(settings.get("general_settings", {}).keys()))
    tag = tag.upper()
    symbol_meta = settings.get("symbol_settings", {}).get(tag)
    if symbol_meta is None:
        raise ValueError(f"Unknown symbol tag: {tag}")

    windows = settings.get("general_settings", {}).get("windows", {})
    if not windows:
        raise ValueError("No windows defined in settings['general_settings']['windows']")

    sim_capital = float(settings.get("simulation_capital", 0))
    start_capital = sim_capital
    realised_pnl = 0.0
    ledger = RamLedger()

    addlog(f"[SIM] Starting simulation for {tag}", verbose_int=1, verbose_state=verbose)

    df = fetch_candles(tag)
    max_note_usdt = settings.get("general_settings", {}).get("max_note_usdt", sim_capital)
    min_note_usdt = settings.get("general_settings", {}).get("minimum_note_size", 0)

    last_buy_tick = {name: float("-inf") for name in windows}
    last_sell_tick: dict[str, int] = {}

    total = len(df)
    with tqdm(total=total, desc="📉 Sim Progress", dynamic_ncols=True) as pbar:
        for tick in range(total):
            offset = total - tick - 1
            price = float(df.iloc[tick]["close"])

            for name, cfg in windows.items():
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
                    total_ticks=total,
                    price=price,
                    sim_capital=sim_capital,
                    last_buy_tick=last_buy_tick,
                    max_note_usdt=max_note_usdt,
                    min_note_usdt=min_note_usdt,
                    verbose=verbose,
                )
                before_pnl = ledger.pnl
                sim_capital, closed = evaluate_sell(
                    ledger=ledger,
                    name=name,
                    cfg=cfg,
                    wave=wave,
                    tick=tick,
                    total_ticks=total,
                    price=price,
                    sim_capital=sim_capital,
                    last_sell_tick=last_sell_tick,
                    verbose=verbose,
                )
                realised_gain = sum(n.get("gain_usdt", 0) for n in closed)
                if realised_gain:
                    realised_pnl += realised_gain

            pbar.update(1)

    last_price = float(df.iloc[-1]["close"])
    open_value = sum(
        n["entry_amount"] * last_price for n in ledger.get_active_notes()
    )
    summarize_simulation(
        ledger=ledger,
        start_capital=start_capital,
        idle_capital=sim_capital,
        realised_pnl=realised_pnl,
        open_value=open_value,
        total_ticks=len(df),
        verbose=verbose,
    )
    print("[SIM] Exiting simulation")

