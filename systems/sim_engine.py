from __future__ import annotations

"""Simple historical simulation engine using wave windows.

This module iterates over historical candle data and performs basic buy/sell
logic based purely on the position of the current price within a windowed wave
range. Configuration is loaded from ``settings/settings.json`` and all trades
are recorded in a lightweight in-memory ledger.
"""

from pathlib import Path
import shutil
import json

from tqdm import tqdm

from systems.scripts.fetch_canles import fetch_candles
from systems.scripts.ledger import Ledger, save_ledger
from systems.scripts.handle_top_of_hour import handle_top_of_hour
from systems.scripts.evaluate_buy import compute_buy_signals
from systems.scripts.evaluate_sell import compute_sell_signals
from systems.scripts.get_window_data import get_wave_window_data_df
from systems.utils.addlog import addlog
from systems.utils.config import load_settings, load_ledger_config, resolve_path


def run_simulation(
    *,
    ledger: str,
    verbose: int = 0,
    window_names: list[str] | None = None,
    output_path: str | None = None,
    sim_logic: str = "sim",
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

    if sim_logic == "live":
        _run_live_logic_simulation(
            ledger=ledger,
            verbose=verbose,
            window_names=window_names,
            output_path=output_path,
        )
        return

    settings = load_settings()
    ledger_config = load_ledger_config(ledger)
    if window_names is not None:
        windows_cfg = ledger_config.get("window_settings", {})
        ledger_config["window_settings"] = {
            name: cfg for name, cfg in windows_cfg.items() if name in window_names
        }

    tag = ledger_config.get("tag", "").upper()

    root = resolve_path("")
    sim_path = (
        Path(output_path)
        if output_path
        else root / "data" / "tmp" / f"simulation_{ledger}.json"
    )
    if sim_path.exists():
        sim_path.unlink()

    windows = ledger_config.get("window_settings", {})
    if not windows:
        raise ValueError("No windows defined for ledger")

    sim_capital = float(settings.get("simulation_capital", 0))
    ledger_obj = Ledger()

    addlog(
        f"[SIM] Starting simulation for {tag}",
        verbose_int=1,
        verbose_state=verbose,
    )

    df = fetch_candles(tag)
    max_note_usdt = settings.get("general_settings", {}).get(
        "max_note_usdt", sim_capital
    )
    min_note_usdt = settings.get("general_settings", {}).get(
        "minimum_note_size", 0
    )

    last_buy_tick = {name: float("-inf") for name in windows}
    buy_cooldown_skips = {name: 0 for name in windows}
    min_roi_gate_hits = 0

    state = {
        "capital": sim_capital,
        "last_buy_tick": last_buy_tick,
        "buy_cooldown_skips": buy_cooldown_skips,
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
                ledger=ledger_obj,
                ledger_config=ledger_config,
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
    save_ledger(ledger, ledger_obj, sim=True, final_tick=final_tick, summary=summary)

    # Copy output to requested path
    default_path = root / "data" / "tmp" / "simulation" / f"{ledger}.json"
    if default_path.exists() and default_path != sim_path:
        sim_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(default_path, sim_path)

    saved_summary = Ledger.load_ledger(ledger, sim=True).get_account_summary(
        final_price
    )
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

    if verbose:
        addlog(
            f"Buy cooldown skips: {state['buy_cooldown_skips']}",
            verbose_int=2,
            verbose_state=verbose,
        )
    addlog(
        f"Min ROI gate hits: {state['min_roi_gate_hits']}",
        verbose_int=1,
        verbose_state=verbose,
    )


def _run_live_logic_simulation(
    *,
    ledger: str,
    verbose: int,
    window_names: list[str] | None,
    output_path: str | None,
) -> None:
    """Run simulation using live trading logic."""

    settings = load_settings()
    ledger_config = load_ledger_config(ledger)
    if window_names is not None:
        windows_cfg = ledger_config.get("window_settings", {})
        ledger_config["window_settings"] = {
            name: cfg for name, cfg in windows_cfg.items() if name in window_names
        }

    tag = ledger_config.get("tag", "").upper()

    root = resolve_path("")
    out_path = root / "data" / "tmp" / "ledger_sim_live.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    windows = ledger_config.get("window_settings", {})
    if not windows:
        raise ValueError("No windows defined for ledger")

    sim_capital = float(settings.get("simulation_capital", 0))
    ledger_obj = Ledger()

    metadata = {
        "tag": tag,
        "last_buy_tick": {name: float("-inf") for name in windows},
        "last_sell_tick": {name: float("-inf") for name in windows},
    }
    ledger_obj.set_metadata(metadata)

    addlog(
        f"[SIM-LIVE] Starting simulation for {tag}",
        verbose_int=1,
        verbose_state=verbose,
    )

    df = fetch_candles(tag)
    max_note_usdt = settings.get("general_settings", {}).get(
        "max_note_usdt", sim_capital
    )
    min_note_usdt = settings.get("general_settings", {}).get(
        "minimum_note_size", 0
    )

    buy_cooldown_skips = {name: 0 for name in windows}
    min_roi_gate_hits = 0

    total = len(df)
    with tqdm(total=total, desc="📉 Sim Progress", dynamic_ncols=True) as pbar:
        for tick in range(total):
            offset = total - tick - 1
            candle = df.iloc[tick].to_dict()
            price = float(candle["close"])

            for name, cfg in windows.items():
                wave = get_wave_window_data_df(
                    df,
                    window=cfg["window_size"],
                    candle_offset=offset,
                )
                if not wave:
                    continue

                signals, skipped = compute_buy_signals(
                    ledger=ledger_obj,
                    name=name,
                    cfg=cfg,
                    wave=wave,
                    tick=tick,
                    price=price,
                    sim_capital=sim_capital,
                    last_buy_tick=metadata["last_buy_tick"],
                    max_note_usdt=max_note_usdt,
                    min_note_usdt=min_note_usdt,
                )
                for sig in signals:
                    note = {
                        "window": sig.name,
                        "entry_tick": tick,
                        "buy_tick": tick,
                        "entry_price": sig.price,
                        "entry_amount": sig.amount,
                        "status": "Open",
                    }
                    ledger_obj.open_note(note)
                    sim_capital -= sig.invest_usdt
                    metadata["last_buy_tick"][name] = tick
                    addlog(
                        f"[SIM-LIVE][BUY] {name} tick {tick} price={price:.6f}",
                        verbose_int=2,
                        verbose_state=verbose,
                    )
                if skipped:
                    buy_cooldown_skips[name] += 1

                sell_signals, roi_skipped = compute_sell_signals(
                    ledger=ledger_obj,
                    name=name,
                    tick=tick,
                    price=price,
                    wave=wave,
                    cfg=cfg,
                    base_sell_cooldown=cfg.get("sell_cooldown", 0),
                    last_sell_tick=metadata["last_sell_tick"],
                )
                min_roi_gate_hits += roi_skipped
                for sig in sell_signals:
                    note = sig.note
                    note["exit_tick"] = tick
                    note["exit_price"] = sig.price
                    note["exit_ts"] = tick
                    note["gain"] = sig.gain
                    note["gain_pct"] = sig.gain_pct
                    note["status"] = "Closed"
                    ledger_obj.close_note(note)
                    sim_capital += note["entry_amount"] * sig.price
                    metadata["last_sell_tick"][name] = tick
                    addlog(
                        f"[SIM-LIVE][SELL] {name} tick {tick} price={price:.6f} gain={sig.gain:.2f}",
                        verbose_int=2,
                        verbose_state=verbose,
                    )

            ledger_obj.set_metadata(metadata)
            pbar.update(1)

    addlog(
        f"[SIM-LIVE] Completed {len(df)} ticks.",
        verbose_int=1,
        verbose_state=verbose,
    )

    final_tick = len(df) - 1 if total else -1
    final_price = float(df.iloc[-1]["close"])
    summary = ledger_obj.get_account_summary(final_price)

    ledger_obj.set_metadata(metadata)
    ledger_data = {
        "open_notes": ledger_obj.get_open_notes(),
        "closed_notes": ledger_obj.get_closed_notes(),
        "metadata": metadata,
        "final_tick": final_tick,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(ledger_data, f, indent=2)

    addlog(
        f"[SIM-LIVE] Final Price: ${summary['final_price']:.2f}",
        verbose_int=1,
        verbose_state=verbose,
    )
    addlog(
        f"[SIM-LIVE] Total Coin Held: {summary['open_coin_amount']:.6f}",
        verbose_int=1,
        verbose_state=verbose,
    )
    addlog(
        f"[SIM-LIVE] Final Value (USD): ${summary['total_value']:.2f}",
        verbose_int=1,
        verbose_state=verbose,
    )
    if verbose:
        addlog(
            f"[SIM-LIVE] Buy cooldown skips: {buy_cooldown_skips}",
            verbose_int=2,
            verbose_state=verbose,
        )
    addlog(
        f"[SIM-LIVE] Min ROI gate hits: {min_roi_gate_hits}",
        verbose_int=1,
        verbose_state=verbose,
    )
