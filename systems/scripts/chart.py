from __future__ import annotations

"""Visualization utilities for simulation trades."""

from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

from .viz_filters import VizFilters


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
    filters = VizFilters.from_settings()
    ax1.plot(df["candle_index"], df["close"], lw=1, label="Close Price", color="blue")
    ax1.set_title(f"Exhaustion Trades\nStart {start_capital}, End {final_value:.2f}")
    ax1.set_xlabel("Candles (Index)")
    ax1.set_ylabel("Price")
    ax1.grid(True)

    for i, r in df.iterrows():
        v = r["angle"]
        if i < angle_lookback or not filters.allow_angle(i):
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

    pb_data = [
        (x, y, s)
        for x, y, s in zip(
            pts["exhaustion_down"]["x"],
            pts["exhaustion_down"]["y"],
            pts["exhaustion_down"]["s"],
        )
        if filters.allow_pressure(s)
    ]
    if pb_data:
        x_p, y_p, s_p = zip(*pb_data)
        ax1.scatter(x_p, y_p, s=s_p, c="green", alpha=0.3, edgecolor="black")

    vb_data = [
        (x, y, s)
        for x, y, s in zip(vol_pts["x"], vol_pts["y"], vol_pts["s"])
        if filters.allow_volatility(s)
    ]
    if vb_data:
        x_v, y_v, s_v = zip(*vb_data)
        ax1.scatter(x_v, y_v, s=s_v, c="red", alpha=0.3, edgecolor="black")

    for t in trades:
        if t["side"] == "BUY":
            ax1.scatter(
                t["idx"],
                t["price"],
                marker="^",
                s=450,  # triple previous marker size
                c="lime",
                edgecolor="black",
                zorder=10,
            )
        elif t["side"] == "SELL":
            ax1.scatter(
                t["idx"],
                t["price"],
                marker="v",
                s=450,  # triple previous marker size
                c="red",
                edgecolor="black",
                zorder=10,
            )

    plt.show()
