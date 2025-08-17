from __future__ import annotations

"""Visualization utilities for simulation runs."""

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
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
        DataFrame containing at least ``close`` prices.
    ledger:
        Mapping with ``open_notes`` and ``closed_notes`` describing trade events.
    tag:
        Identifier used for the output filename.
    outfile:
        Optional explicit output path. Defaults to ``data/tmp/{tag}_viz.png``.
    """

    df = candles.copy()
    x_values = list(range(len(df)))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x_values, df["close"], color="black", linewidth=1)

    buys_x: List[int] = []
    buys_y: List[float] = []
    sells_x: List[int] = []
    sells_y: List[float] = []
    flat_sells_x: List[int] = []
    flat_sells_y: List[float] = []
    jackpot_x: List[int] = []
    jackpot_y: List[float] = []

    notes = ledger.get("open_notes", []) + ledger.get("closed_notes", [])
    buy_count = len(notes)

    for note in notes:
        entry_idx = note.get("entry_idx")
        entry_price = note.get("entry_price")
        if entry_idx is not None and entry_price is not None:
            buys_x.append(entry_idx)
            buys_y.append(entry_price)
            if note.get("kind") == "jackpot":
                jackpot_x.append(entry_idx)
                jackpot_y.append(entry_price)

    pressure_sell_count = 0
    flat_sell_count = 0
    jackpot_note_ids = {n.get("id") for n in notes if n.get("kind") == "jackpot"}

    for note in ledger.get("closed_notes", []):
        exit_idx = note.get("exit_idx")
        exit_price = note.get("exit_price")
        reason = note.get("reason")
        if exit_idx is None or exit_price is None:
            continue
        if reason == "PRESSURE_SELL":
            sells_x.append(exit_idx)
            sells_y.append(exit_price)
            pressure_sell_count += 1
        elif reason == "FLAT_SELL":
            flat_sells_x.append(exit_idx)
            flat_sells_y.append(exit_price)
            flat_sell_count += 1
        if note.get("kind") == "jackpot":
            jackpot_x.append(exit_idx)
            jackpot_y.append(exit_price)

    ax.scatter(buys_x, buys_y, color="green", marker="o", label=f"BUY (count={buy_count})")
    ax.scatter(sells_x, sells_y, color="red", marker="o", label=f"SELL (count={pressure_sell_count})")
    ax.scatter(flat_sells_x, flat_sells_y, color="orange", marker="o", label=f"FLAT_SELL (count={flat_sell_count})")
    if jackpot_x:
        ax.scatter(
            jackpot_x,
            jackpot_y,
            color="blue",
            marker="o",
            label=f"JACKPOT (count={len(jackpot_note_ids)})",
        )

    ax.set_title("Simulation")
    ax.set_xlabel("Candles (Index)")
    ax.set_ylabel("Price")
    ax.legend()

    outfile = outfile or f"data/tmp/{tag}_viz.png"
    out_path = Path(outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.show()
    plt.close(fig)
    print(f"[VIZ] Saved plot to {out_path}")

