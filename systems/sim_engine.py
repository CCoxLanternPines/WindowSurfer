from __future__ import annotations

"""Very small historical simulation engine."""

import re
from datetime import timedelta
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .scripts import evaluate_buy, evaluate_sell


# Step size (candles) for slope updates
# If < 1, treated as fraction of dataset length
DEFAULT_BOTTOM_WINDOW = 0.1


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

    total_candles = len(df)
    if DEFAULT_BOTTOM_WINDOW < 1:
        BOTTOM_WINDOW = max(1, int(total_candles * DEFAULT_BOTTOM_WINDOW))
    else:
        BOTTOM_WINDOW = int(DEFAULT_BOTTOM_WINDOW)

    print(
        f"[SIM] Using BOTTOM_WINDOW={BOTTOM_WINDOW} (derived from {DEFAULT_BOTTOM_WINDOW})"
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

    # Half-window slope prediction + snapback exploration
    predicted_angles = [np.nan] * len(df)
    snapback_conf = [np.nan] * len(df)
    matches = 0
    total = 0

    for i in range(0, len(df), BOTTOM_WINDOW):
        end = min(i + BOTTOM_WINDOW, len(df))
        mid = i + BOTTOM_WINDOW // 2

        # First half slope
        y1 = df["close"].iloc[i:mid].values
        x1 = np.arange(len(y1))
        angle1 = None
        if len(y1) > 1:
            m1, _ = np.polyfit(x1, y1, 1)
            angle1 = np.tanh(m1)
            predicted_angles[mid:end] = [angle1] * (end - mid)

        # Second half slope (projected forward)
        y2 = df["close"].iloc[mid:end].values
        x2 = np.arange(len(y2))
        if len(y2) > 1:
            m2, _ = np.polyfit(x2, y2, 1)
            angle2 = np.tanh(m2)
            predicted_angles[end : end + BOTTOM_WINDOW] = [angle2] * min(
                BOTTOM_WINDOW, len(df) - end
            )

            if angle1 is not None:
                # Snapback confidence high if directions oppose
                snapback = abs(angle1 - angle2) / 2
                snapback_conf[mid:end] = [snapback] * (end - mid)
                total += 1
                if np.sign(angle1) == np.sign(angle2):
                    matches += 1

    if total > 0:
        acc = matches / total * 100
        print(f"[SIM] Half-window slope directional accuracy: {acc:.1f}%")

    df["predicted_angle"] = predicted_angles
    df["snapback_conf"] = snapback_conf

    state: Dict[str, Any] = {}
    for _, candle in df.iterrows():
        evaluate_buy.evaluate_buy(candle.to_dict(), state)
        evaluate_sell.evaluate_sell(candle.to_dict(), state)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df["candle_index"], df["close"], label="Close Price", color="blue")
    ax1.plot(
        df["candle_index"],
        df["bottom_slope"],
        label=f"Slope Line ({BOTTOM_WINDOW})",
        color="black",
        linewidth=2,
        drawstyle="steps-post",
    )
    ax1.set_ylabel("Price")
    ax1.set_xlabel("Candles (Index)")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(
        df["candle_index"],
        df["predicted_angle"],
        label="Predicted Slope Angle (half-window)",
        color="green",
        drawstyle="steps-post",
    )
    ax2.plot(
        df["candle_index"],
        df["snapback_conf"],
        label="Snapback Confidence",
        color="orange",
        linestyle="--",
        drawstyle="steps-post",
    )
    ax2.set_ylim(-1, 1)
    ax2.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax2.set_ylabel("Slope Angle / Confidence")
    ax2.legend(loc="lower right")

    plt.title("SOLUSD Discovery Simulation")
    plt.grid(True)
    plt.show()
