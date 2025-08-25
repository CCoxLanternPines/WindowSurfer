from __future__ import annotations

"""Visualization utilities for simulation trades."""

from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def plot_trades(
    df: pd.DataFrame,
    pts: Dict[str, Dict[str, List[float]]],
    vol_pts: Dict[str, List[float]],
    trades: List[Dict[str, float]],
    start_capital: float,
    final_value: float,
    angle_lookback: int = 48,
) -> None:
    """Plot price, exhaustion points, volatility, and trade markers."""
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df["candle_index"], df["close"], lw=1, label="Close Price", color="blue")
    ax1.set_title(f"Exhaustion Trades\nStart {start_capital}, End {final_value:.2f}")
    ax1.set_xlabel("Candles (Index)")
    ax1.set_ylabel("Price")
    ax1.grid(True)

    for i, r in df.iterrows():
        v = r["angle"]
        if i < angle_lookback:
            continue
        if v > 0.05:
            color = "orange"
        elif v < -0.05:
            color = "purple"
        else:
            color = "gray"
        x0, y0 = r["candle_index"], r["close"]
        x1 = x0 + 5
        y1 = y0 + v * 5
        ax1.plot([x0, x1], [y0, y1], color=color, lw=1.5, alpha=0.7)

    ax1.scatter(
        pts["exhaustion_down"]["x"],
        pts["exhaustion_down"]["y"],
        s=pts["exhaustion_down"]["s"],
        c="green",
        alpha=0.3,
        edgecolor="black",
    )

    ax1.scatter(
        vol_pts["x"],
        vol_pts["y"],
        s=vol_pts["s"],
        c="red",
        alpha=0.3,
        edgecolor="black",
    )

    for t in trades:
        if t["side"] == "BUY":
            ax1.scatter(
                t["idx"],
                t["price"],
                marker="^",
                s=150,
                c="lime",
                edgecolor="black",
                zorder=10,
            )
        elif t["side"] == "SELL":
            ax1.scatter(
                t["idx"],
                t["price"],
                marker="v",
                s=150,
                c="red",
                edgecolor="black",
                zorder=10,
            )

    plt.show()
