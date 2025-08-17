from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd


def plot_pressure(candles: pd.DataFrame, ledger: Dict[str, Any], outfile: str | None = None) -> None:
    """Plot price candles with buy/sell markers based on ledger events."""

    df = candles.copy()

    # Determine a timestamp column if present for the x-axis
    ts_col = None
    for c in ("timestamp", "time", "date"):
        if c in df.columns:
            ts_col = c
            break

    if ts_col:
        x_values = pd.to_datetime(df[ts_col], unit="s")
    else:
        x_values = df.index

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x_values, df["close"], color="black", linewidth=1, label="Close")

    buys_x: list[Any] = []
    buys_y: list[float] = []
    pressure_sells_x: list[Any] = []
    pressure_sells_y: list[float] = []
    flat_sells_x: list[Any] = []
    flat_sells_y: list[float] = []
    jackpot_x: list[Any] = []
    jackpot_y: list[float] = []

    notes = ledger.get("open_notes", []) + ledger.get("closed_notes", [])
    for note in notes:
        entry_idx = note.get("entry_idx")
        entry_ts = note.get("created_ts")
        entry_price = note.get("entry_price")

        if ts_col and entry_ts is not None:
            x_entry = pd.to_datetime(entry_ts, unit="s")
        elif entry_idx is not None:
            x_entry = x_values[entry_idx] if len(x_values) > entry_idx else entry_idx
        else:
            x_entry = None
        if x_entry is not None and entry_price is not None:
            buys_x.append(x_entry)
            buys_y.append(entry_price)
            if note.get("kind") == "jackpot":
                jackpot_x.append(x_entry)
                jackpot_y.append(entry_price)

        exit_idx = note.get("exit_idx")
        exit_ts = note.get("exit_ts")
        exit_price = note.get("exit_price")
        if exit_price is not None:
            if ts_col and exit_ts is not None:
                x_exit = pd.to_datetime(exit_ts, unit="s")
            elif exit_idx is not None:
                x_exit = x_values[exit_idx] if len(x_values) > exit_idx else exit_idx
            else:
                x_exit = None
            reason = note.get("reason")
            if x_exit is not None:
                if reason == "PRESSURE_SELL":
                    pressure_sells_x.append(x_exit)
                    pressure_sells_y.append(exit_price)
                elif reason == "FLAT_SELL":
                    flat_sells_x.append(x_exit)
                    flat_sells_y.append(exit_price)
                if note.get("kind") == "jackpot":
                    jackpot_x.append(x_exit)
                    jackpot_y.append(exit_price)

    ax.scatter(buys_x, buys_y, color="green", marker="o", label="BUY")
    ax.scatter(pressure_sells_x, pressure_sells_y, color="red", marker="o", label="PRESSURE_SELL")
    ax.scatter(flat_sells_x, flat_sells_y, color="orange", marker="o", label="FLAT_SELL")
    if jackpot_x:
        ax.scatter(jackpot_x, jackpot_y, color="blue", marker="o", label="JACKPOT")

    ax.set_title("Pressure Simulation")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()

    outfile = outfile or "data/tmp/sim_plot.png"
    out_path = Path(outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[PLOT] Saved plot to {out_path}")
