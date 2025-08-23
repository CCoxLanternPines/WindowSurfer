from __future__ import annotations

"""Exhaustion brain producing red/green cluster signals."""

from collections import deque
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..sim_engine import (
    WINDOW_SIZE,
    WINDOW_STEP,
    CLUSTER_WINDOW,
    BASE_SIZE,
    SCALE_POWER,
    multi_window_vote,
)


def run(df: pd.DataFrame, viz: bool):
    """Return exhaustion signals, optionally plotting them."""
    signals: List[Dict[str, float]] = []
    recent_buys = deque(maxlen=CLUSTER_WINDOW)
    recent_sells = deque(maxlen=CLUSTER_WINDOW)

    xr, yr, sr = [], [], []
    xg, yg, sg = [], [], []

    for t in range(WINDOW_SIZE - 1, len(df), WINDOW_STEP):
        x = int(df["candle_index"].iloc[t])
        y = float(df["close"].iloc[t])
        decision, _, _ = multi_window_vote(df, t, window_sizes=[8, 12, 24, 48])
        if decision == 1:
            recent_buys.append(t)
            cluster_strength = sum(1 for idx in recent_buys if t - idx <= CLUSTER_WINDOW)
            size = BASE_SIZE * (cluster_strength ** SCALE_POWER)
            xr.append(x)
            yr.append(y)
            sr.append(size)
            signals.append({"candle_index": x, "price": y, "direction": "sell", "size": size})
        elif decision == -1:
            recent_sells.append(t)
            cluster_strength = sum(1 for idx in recent_sells if t - idx <= CLUSTER_WINDOW)
            size = BASE_SIZE * (cluster_strength ** SCALE_POWER)
            xg.append(x)
            yg.append(y)
            sg.append(size)
            signals.append({"candle_index": x, "price": y, "direction": "buy", "size": size})

    if viz:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df["candle_index"], df["close"], lw=1, label="Close Price", color="blue")
        ax.scatter(xr, yr, s=sr, c="red", zorder=6)
        ax.scatter(xg, yg, s=sg, c="green", zorder=6)
        ax.set_title("Price with Exhaustion (Brain)")
        ax.set_xlabel("Candles (Index)")
        ax.set_ylabel("Price")
        ax.grid(True)
        plt.show()

    return signals


def summarize(signals: List[Dict[str, float]], df: pd.DataFrame):
    """Compute basic statistics for the signal list."""
    count = len(signals)
    indices = [s["candle_index"] for s in signals]
    avg_gap = int(np.mean(np.diff(indices))) if len(indices) > 1 else 0
    up = sum(1 for s in signals if s["direction"] == "sell")
    down = sum(1 for s in signals if s["direction"] == "buy")
    total = up + down
    if total:
        pct = int(round(100 * up / total))
        direction = "uptrend" if up >= down else "downtrend"
    else:
        pct = 0
        direction = "flat"
    return {"count": count, "avg_gap": avg_gap, "slope_bias": f"{direction} {pct}%"}
