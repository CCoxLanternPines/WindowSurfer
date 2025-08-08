from __future__ import annotations

"""Simple historical simulation engine using wave windows.

This module iterates over historical candle data and performs basic buy/sell
logic based purely on the position of the current price within a windowed wave
range. Configuration is loaded from ``settings/settings.json`` and all trades
are recorded in a lightweight in-memory ledger.
"""

from pathlib import Path
import shutil

from tqdm import tqdm

from systems.scripts.fetch_candles import fetch_candles
from systems.scripts.ledger import Ledger, save_ledger
from systems.scripts.get_window_data import get_wave_window_data_df
from systems.scripts.evaluate_buy import evaluate_buy
from systems.scripts.evaluate_sell import evaluate_sell
from systems.scripts.execution_handler import execute_buy, execute_sell
from systems.utils.addlog import addlog
from systems.utils.config import load_settings, load_ledger_config, resolve_path
from systems.utils.symbols import resolve_asset, resolve_tag


def run_simulation(
    *,
    ledger: str,
    verbose: int = 0,
    window_names: list[str] | None = None,
    output_path: str | None = None,
) -> None:
    """Run a historical simulation for ``ledger``.

    Parameters
    ----------
    ledger:
        Name of the ledger defined in ``settings.json``.
    window_names:
        Optional subset of window names to simulate. If ``None``, all windows
        for the ledger are used.
    output_path:
        Optional path to write the resulting ledger JSON. Defaults to
        ``data/tmp/simulation_<ledger>.json``.
    """

    settings = load_settings()
    ledger_config = load_ledger_config(ledger)
    if window_names is not None:
        windows_cfg = ledger_config.get("window_settings", {})
        ledger_config["window_settings"] = {
            name: cfg for name, cfg in windows_cfg.items() if name in window_names
        }

    tag = resolve_tag(ledger_config)
    asset = resolve_asset(ledger_config)

    root = resolve_path("")
    default_sim_path = root / "data" / "tmp" / "simulation" / f"{asset}.json"
    sim_path = Path(output_path) if output_path else default_sim_path
    if sim_path.exists():
        sim_path.unlink()

    windows = ledger_config.get("window_settings", {})
    if not windows:
        raise ValueError("No windows defined for ledger")

    sim_capital = float(settings.get("simulation_capital", 0))
    ledger_obj = Ledger()

    addlog(
        f"[SIM] Starting simulation for {asset} ({tag})",
        verbose_int=1,
        verbose_state=verbose,
    )

    df = fetch_candles(asset=asset)
    max_note_usdt = settings.get("general_settings", {}).get("max_note_usdt", sim_capital)
    min_note_usdt = settings.get("general_settings", {}).get("minimum_note_size", 0)

    state = {"capital": sim_capital}
    cooldowns = {name: {"buy": 0, "sell": 0} for name in windows}
    state["cooldowns"] = cooldowns
    metadata = {"asset": asset, "tag": tag, "last_id": 0}
    ledger_obj.set_metadata(metadata)

    total = len(df)
    with tqdm(total=total, desc="📉 Sim Progress", dynamic_ncols=True) as pbar:
        for tick in range(total):
            offset = total - tick - 1
            candle = df.iloc[tick].to_dict()
            price = float(candle.get("close", 0.0))

            for cd in cooldowns.values():
                if cd["buy"] > 0:
                    cd["buy"] -= 1
                if cd["sell"] > 0:
                    cd["sell"] -= 1

            for name, cfg in windows.items():
                wave = get_wave_window_data_df(
                    df,
                    window=cfg["window_size"],
                    candle_offset=offset,
                )
                if not wave:
                    continue

                buy_signal = evaluate_buy(
                    state=state,
                    ledger=ledger_obj,
                    strategy=name,
                    cfg=cfg,
                    wave=wave,
                    tick=tick,
                    price=price,
                    cooldowns=cooldowns,
                    max_note_usdt=max_note_usdt,
                    min_note_usdt=min_note_usdt,
                    verbose=verbose,
                )
                if buy_signal:
                    execute_buy(
                        symbol=tag,
                        price=buy_signal["price"],
                        amount_usd=buy_signal["amount_usd"],
                        ledger_name=ledger,
                        ledger=ledger_obj,
                        tick=tick,
                        strategy=name,
                        cooldowns=cooldowns,
                        settings=cfg,
                        state=state,
                        sim=True,
                        verbose=verbose,
                    )

                sell_signals = evaluate_sell(
                    state=state,
                    ledger=ledger_obj,
                    strategy=name,
                    cfg=cfg,
                    wave=wave,
                    tick=tick,
                    price=price,
                    cooldowns=cooldowns,
                    verbose=verbose,
                )
                for sig in sell_signals:
                    execute_sell(
                        symbol=tag,
                        coin_amount=sig["note"]["entry_amount"],
                        price=sig["price"],
                        ledger_name=ledger,
                        ledger=ledger_obj,
                        tick=tick,
                        note=sig["note"],
                        cooldowns=cooldowns,
                        settings=cfg,
                        state=state,
                        sim=True,
                        verbose=verbose,
                    )

            pbar.update(1)

    addlog(
        f"[SIM] Completed {len(df)} ticks.",
        verbose_int=1,
        verbose_state=verbose,
    )

    final_tick = len(df) - 1 if total else -1
    final_price = float(df.iloc[-1]["close"])
    summary = ledger_obj.get_account_summary(final_price)

    addlog(
        f"[DEBUG] Final tick: {final_tick}",
        verbose_int=3,
        verbose_state=verbose,
    )
    save_ledger(asset, ledger_obj, sim=True, final_tick=final_tick, summary=summary)

    # Copy output to requested path
    if default_sim_path.exists() and default_sim_path != sim_path:
        sim_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(default_sim_path, sim_path)

    saved_summary = Ledger.load_ledger(asset, sim=True).get_account_summary(final_price)
    if (
        saved_summary["closed_notes"] != summary["closed_notes"]
        or saved_summary["realized_gain"] != summary["realized_gain"]
    ):
        addlog(
            "[WARN] Summary/ledger mismatch: "
            f"closed_notes {summary['closed_notes']} vs {saved_summary['closed_notes']}, "
            f"realized_gain {summary['realized_gain']:.2f} vs {saved_summary['realized_gain']:.2f}",
            verbose_int=1,
            verbose_state=verbose,
        )

    addlog(
        f"Final Price: ${summary['final_price']:.2f}",
        verbose_int=1,
        verbose_state=verbose,
    )
    addlog(
        f"Total Coin Held: {summary['open_coin_amount']:.6f}",
        verbose_int=1,
        verbose_state=verbose,
    )
    addlog(
        f"Final Value (USD): ${summary['total_value']:.2f}",
        verbose_int=1,
        verbose_state=verbose,
    )

    capital = settings.get("simulation_capital", 0.0)
    addlog(
        f"[SUMMARY] Simulated Capital: ${capital:.2f} | "
        f"Realized Gain: ${summary['realized_gain']:.2f} | "
        f"Final Value: ${capital + summary['realized_gain']:.2f}",
        verbose_int=0,
        verbose_state=0,
    )

    # Simulation summary logged above; no extra counters in new pipeline.

