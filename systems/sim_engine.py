from __future__ import annotations

"""Very small historical simulation engine."""

import re
from datetime import timedelta

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd


# Box visualization knobs
WINDOW_SIZE = 150   # candles per box
STEP_SIZE = 150     # rolling step


def parse_timeframe(tf: str) -> timedelta | None:
    match = re.match(r"(\d+)([dhmw])", tf)
    if not match:
        return None
    val, unit = int(match.group(1)), match.group(2)
    if unit == "d":
        return timedelta(days=val)
    if unit == "w":
        return timedelta(weeks=val)
    if unit == "m":
        return timedelta(days=30 * val)  # rough month
    if unit == "h":
        return timedelta(hours=val)
    return None


def run_simulation(*, timeframe: str = "1m") -> None:
    """Run a simple simulation over SOLUSD candles."""
    file_path = "data/sim/SOLUSD_1h.csv"
    df = pd.read_csv(file_path)

    if timeframe:
        delta = parse_timeframe(timeframe)
        if delta:
            cutoff = (pd.Timestamp.utcnow().tz_localize(None) - delta).timestamp()
            df = df[df["timestamp"] >= cutoff]

    df = df.reset_index(drop=True)
    df["candle_index"] = range(len(df))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["candle_index"], df["close"], color="blue", label="Close Price")

    for start in range(0, len(df) - WINDOW_SIZE, STEP_SIZE):
        end = start + WINDOW_SIZE
        sub = df.iloc[start:end]
        low = sub["close"].min()
        high = sub["close"].max()

        rect = patches.Rectangle(
            (start, low),
            WINDOW_SIZE,
            high - low,
            linewidth=1.5,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)

    ax.set_title("Rolling Window Box Visualization")
    ax.set_xlabel("Candles (Index)")
    ax.set_ylabel("Price")
    ax.legend(loc="upper left")
    ax.grid(True)
    plt.show()

