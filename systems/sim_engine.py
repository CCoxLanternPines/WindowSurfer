from __future__ import annotations

"""Very small historical simulation engine."""

import re
from datetime import timedelta
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .scripts import evaluate_buy, evaluate_sell


# === Regime Detection Settings ===
SHORT_MA = 10   # short moving average lookback (candles)
LONG_MA = 50    # long moving average lookback (candles)

# Step size (candles) for slope updates
BOTTOM_WINDOW = 10


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
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

    if timeframe:
        delta = parse_timeframe(timeframe)
        if delta:
            cutoff = pd.Timestamp.utcnow().tz_localize(None) - delta
            df = df[df["timestamp"] >= cutoff]

    df["short"] = df["close"].rolling(window=SHORT_MA, min_periods=1).mean()
    df["long"] = df["close"].rolling(window=LONG_MA, min_periods=1).mean()
    df["delta"] = df["short"] - df["long"]

    # scale delta into the price range
    scale = (df["close"].max() - df["close"].min()) / (
        df["delta"].max() - df["delta"].min()
    )
    df["norm_delta"] = (
        (df["delta"] - df["delta"].min()) * scale + df["close"].min()
    )

    # Stepwise slope calculation
    slopes = [np.nan] * len(df)
    slope_angles = [np.nan] * len(df)
    last_value = df["close"].iloc[0]

    for i in range(0, len(df), BOTTOM_WINDOW):
        end = min(i + BOTTOM_WINDOW, len(df))
        y = df["close"].iloc[i:end].values
        x = np.arange(len(y))
        if len(y) > 1:
            m, b = np.polyfit(x, y, 1)
            fitted = last_value + m * np.arange(len(y))
            slopes[i:end] = fitted
            last_value = fitted[-1]
            slope_val = np.tanh(m)
            slope_angles[i:end] = [slope_val] * len(y)
        else:
            slopes[i:end] = [last_value] * len(y)
            slope_angles[i:end] = [0] * len(y)

    df["bottom_slope"] = slopes
    df["slope_angle"] = slope_angles

    state: Dict[str, Any] = {}
    for _, candle in df.iterrows():
        evaluate_buy.evaluate_buy(candle.to_dict(), state)
        evaluate_sell.evaluate_sell(candle.to_dict(), state)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df["timestamp"], df["close"], label="Close Price", color="blue")
    ax1.plot(
        df["timestamp"],
        df["bottom_slope"],
        label=f"Slope Line ({BOTTOM_WINDOW})",
        color="black",
        linewidth=2,
        drawstyle="steps-post",
    )
    ax1.set_ylabel("Price")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(
        df["timestamp"],
        df["slope_angle"],
        label="Slope Angle [-1,1]",
        color="green",
        alpha=0.6,
        drawstyle="steps-post",
    )
    ax2.set_ylim(-1, 1)
    ax2.set_ylabel("Slope Angle")
    ax2.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax2.legend(loc="lower right")

    plt.title("SOLUSD Discovery Simulation")
    plt.grid(True)
    plt.show()
