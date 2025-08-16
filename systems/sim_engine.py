from __future__ import annotations

"""Very small historical simulation engine."""

import re
from datetime import timedelta

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Box visualization knobs
WINDOW_SIZE = 150   # candles per box
STEP_SIZE = 150     # rolling step

# Reversal detection knobs
SLOPE_WINDOW = 8        # candles for slope calc
LOCAL_WINDOW = 12       # lookback for local high/low
VOL_MULT = 1.5          # min volume spike multiple
REVERSAL_PCT = 0.01     # 1% move to confirm reversal


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


def detect_reversals(df: pd.DataFrame) -> list[tuple[int, float, str]]:
    """Detect reversal points using slope, local extrema and volume."""
    reversals: list[tuple[int, float, str]] = []
    last_extreme_price: float | None = None
    prev_slope: float | None = None

    for i in range(SLOPE_WINDOW, len(df)):
        window = df["close"].iloc[i - SLOPE_WINDOW + 1 : i + 1]
        x = np.arange(len(window))
        slope = float(np.polyfit(x, window, 1)[0])

        if prev_slope is not None and prev_slope * slope < 0:
            idx = i - 1
            price = float(df.iloc[idx]["close"])
            volume = float(df.iloc[idx]["volume"])

            direction = "peak" if prev_slope > 0 else "trough"

            start = max(0, idx - LOCAL_WINDOW + 1)
            closes = df["close"].iloc[start : idx + 1]
            if direction == "peak" and price < closes.max():
                pass
            elif direction == "trough" and price > closes.min():
                pass
            else:
                vol_start = max(0, idx - SLOPE_WINDOW)
                vol_avg = df["volume"].iloc[vol_start:idx].mean()
                if vol_avg > 0 and volume >= VOL_MULT * vol_avg:
                    if (
                        last_extreme_price is None
                        or abs(price - last_extreme_price) / last_extreme_price
                        >= REVERSAL_PCT
                    ):
                        reversals.append((idx, price, direction))
                        last_extreme_price = price

        prev_slope = slope

    return reversals


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
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df["candle_index"], df["close"], color="blue", label="Close Price")

    reversals = detect_reversals(df)
    if reversals:
        ax1.scatter(
            [],
            [],
            marker="v",
            color="green",
            edgecolor="black",
            label=f"Reversals ({len(reversals)})",
        )
        for idx, price, direction in reversals:
            y = price * (1 + REVERSAL_PCT) if direction == "peak" else price * (1 - REVERSAL_PCT)
            marker = "v" if direction == "peak" else "^"
            ax1.scatter(
                idx,
                y,
                marker=marker,
                color="green",
                edgecolor="black",
                zorder=5,
            )

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
        ax1.add_patch(rect)

    ax1.set_title("Rolling Window Box Visualization")
    ax1.set_xlabel("Candles (Index)")
    ax1.set_ylabel("Price")
    ax1.legend(loc="upper left")
    ax1.grid(True)
    plt.show()

