from __future__ import annotations

"""Visualization utilities for simulation runs."""

from pathlib import Path
from typing import Any, Dict

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd


def plot_viz(
    candles: pd.DataFrame,
    ledger: Dict[str, Any],
    tag: str,
    outfile: str | None = None,
) -> None:
    """Plot candle prices with buy/sell markers.

    Parameters
    ----------
    candles:
        DataFrame containing at least ``open``, ``high``, ``low``, ``close`` and a timestamp column.
    ledger:
        Mapping with ``open_notes`` and ``closed_notes`` describing trade events.
    tag:
        Identifier used for the output filename.
    outfile:
        Optional explicit output path. Defaults to ``data/tmp/{tag}_viz.png``.
    """

    df = candles.copy()

    # Determine a timestamp column if present for the x-axis
    ts_col = None
    for c in ("timestamp", "time", "date"):
        if c in df.columns:
            ts_col = c
            break

    if ts_col:
        df["x"] = pd.to_datetime(df[ts_col], unit="s").map(mdates.date2num)
    else:
        df["x"] = range(len(df))

    fig, ax = plt.subplots(figsize=(12, 6))

    # Determine candle body width
    if len(df["x"]) > 1:
        width = (df["x"].iloc[1] - df["x"].iloc[0]) * 0.8
    else:
        width = 0.6

    for _, row in df.iterrows():
        color = "green" if row["close"] >= row["open"] else "red"
        ax.plot([row["x"], row["x"]], [row["low"], row["high"]], color="black", linewidth=0.5)
        lower = min(row["open"], row["close"])
        height = abs(row["close"] - row["open"])
        rect = Rectangle((row["x"] - width / 2, lower), width, height, color=color)
        ax.add_patch(rect)

    buys_x: list[float] = []
    buys_y: list[float] = []
    sells_x: list[float] = []
    sells_y: list[float] = []
    jackpot_x: list[float] = []
    jackpot_y: list[float] = []

    notes = ledger.get("open_notes", []) + ledger.get("closed_notes", [])
    for note in notes:
        entry_idx = note.get("entry_idx")
        entry_ts = note.get("created_ts")
        entry_price = note.get("entry_price")

        if ts_col and entry_ts is not None:
            x_entry = mdates.date2num(pd.to_datetime(entry_ts, unit="s"))
        elif entry_idx is not None and len(df["x"]) > entry_idx:
            x_entry = df["x"].iloc[entry_idx]
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
                x_exit = mdates.date2num(pd.to_datetime(exit_ts, unit="s"))
            elif exit_idx is not None and len(df["x"]) > exit_idx:
                x_exit = df["x"].iloc[exit_idx]
            else:
                x_exit = None
            if x_exit is not None:
                sells_x.append(x_exit)
                sells_y.append(exit_price)
                if note.get("kind") == "jackpot":
                    jackpot_x.append(x_exit)
                    jackpot_y.append(exit_price)

    ax.scatter(buys_x, buys_y, color="green", marker="o", label="BUY")
    ax.scatter(sells_x, sells_y, color="red", marker="o", label="SELL")
    if jackpot_x:
        ax.scatter(jackpot_x, jackpot_y, color="blue", marker="o", label="JACKPOT")

    ax.set_title("Simulation")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()

    if ts_col:
        ax.xaxis_date()
        fig.autofmt_xdate()

    outfile = outfile or f"data/tmp/{tag}_viz.png"
    out_path = Path(outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.show()
    plt.close(fig)
    print(f"[VIZ] Saved plot to {out_path}")

