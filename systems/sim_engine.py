from __future__ import annotations

"""Very small historical simulation engine."""

import re
from datetime import timedelta

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Box visualization knobs
WINDOW_SIZE = 24   # candles per box
STEP_SIZE = 24     # rolling step
BAR_COLOR = "orange"
BAR_ALPHA = 0.7

# Reversal detection knobs
SLOPE_WINDOW = 8        # candles for slope calc
LOCAL_WINDOW = 12       # lookback for local high/low
VOL_MULT = 1.5          # min volume spike multiple
REVERSAL_PCT = 0.01     # 1% move to confirm reversal

# Future trend prediction knobs
SLOPE_THRESHOLD = 0.0       # minimum slope magnitude for signal
VOLATILITY_THRESHOLD = 0.02 # min % range of trailing window
VOLUME_THRESHOLD = 1.5      # volume spike multiple vs avg
CONFIDENCE_THRESHOLD = 0.6  # min confidence for filtered accuracy
LOOKAHEAD_HOURS = 24        # prediction horizon


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

    predictions: list[dict[str, float | str]] = []
    bars: list[tuple[int, int, float]] = []
    arrows: list[tuple[int, float, str, float]] = []

    for start in range(0, len(df) - WINDOW_SIZE - LOOKAHEAD_HOURS, STEP_SIZE):
        end = start + WINDOW_SIZE
        sub = df.iloc[start:end]
        low = float(sub["low"].min()) if "low" in sub else float(sub["close"].min())
        high = float(sub["high"].max()) if "high" in sub else float(sub["close"].max())

        rect = patches.Rectangle(
            (start, low),
            WINDOW_SIZE,
            high - low,
            linewidth=1.5,
            edgecolor="red",
            facecolor="none",
        )
        ax1.add_patch(rect)
        level = float(df.iloc[start]["close"])
        bars.append((start, end, level))

        # --- Feature extraction at window entry ---
        slope_window = df["close"].iloc[max(0, start - SLOPE_WINDOW + 1) : start + 1]
        x = np.arange(len(slope_window))
        slope = float(np.polyfit(x, slope_window, 1)[0]) if len(slope_window) > 1 else 0.0

        vol_window = df.iloc[max(0, start - SLOPE_WINDOW + 1) : start + 1]
        volatility = 0.0
        if len(vol_window):
            vol_high = float(vol_window["high"].max()) if "high" in vol_window else float(vol_window["close"].max())
            vol_low = float(vol_window["low"].min()) if "low" in vol_window else float(vol_window["close"].min())
            volatility = (vol_high - vol_low) / level if level else 0.0
            vol_avg = float(vol_window["volume"].mean())
        else:
            vol_avg = 0.0
        volume = float(df.iloc[start]["volume"]) if "volume" in df else 0.0
        volume_delta = (volume / vol_avg) if vol_avg else 0.0

        position = (level - low) / (high - low) if high > low else 0.5

        # --- Prediction rule ---
        direction = "UP" if slope >= 0 else "DOWN"
        slope_pass = abs(slope) >= SLOPE_THRESHOLD if SLOPE_THRESHOLD > 0 else True
        vol_pass = volatility >= VOLATILITY_THRESHOLD if VOLATILITY_THRESHOLD > 0 else True
        volm_pass = volume_delta >= VOLUME_THRESHOLD if VOLUME_THRESHOLD > 0 else True
        confidence = (int(slope_pass) + int(vol_pass) + int(volm_pass)) / 3.0

        future_idx = start + LOOKAHEAD_HOURS
        future_price = float(df.iloc[future_idx]["close"])
        actual = "UP" if future_price > level else "DOWN"

        predictions.append(
            {
                "predicted": direction,
                "actual": actual,
                "confidence": confidence,
                "slope": slope,
                "volatility": volatility,
                "volume_delta": volume_delta,
                "position": position,
            }
        )
        arrows.append((start, level, direction, confidence))

    ax1.plot([], [], color=BAR_COLOR, alpha=BAR_ALPHA, linewidth=1.5, label="Window Entry")
    for start, end, level in bars:
        ax1.hlines(
            y=level,
            xmin=start,
            xmax=end,
            colors=BAR_COLOR,
            linewidth=1.5,
            alpha=BAR_ALPHA,
        )

    for idx, price, direction, conf in arrows:
        marker = "^" if direction == "UP" else "v"
        color = "green" if direction == "UP" else "red"
        alpha = 1.0 if conf >= CONFIDENCE_THRESHOLD else 0.3
        ax1.scatter(idx, price, marker=marker, color=color, alpha=alpha, zorder=5)

    ax1.set_title("Rolling Window Box Visualization")
    ax1.set_xlabel("Candles (Index)")
    ax1.set_ylabel("Price")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    total = len(predictions)
    correct = sum(1 for p in predictions if p["predicted"] == p["actual"])
    raw_acc = 100 * correct / total if total else 0.0
    filt = [p for p in predictions if p["confidence"] >= CONFIDENCE_THRESHOLD]
    filt_total = len(filt)
    filt_correct = sum(1 for p in filt if p["predicted"] == p["actual"])
    filt_acc = 100 * filt_correct / filt_total if filt_total else 0.0
    print(
        f"Raw Accuracy: {raw_acc:.2f}% | Filtered Accuracy: {filt_acc:.2f}% ({filt_total}/{total})"
    )

    plt.show()

