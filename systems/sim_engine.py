from __future__ import annotations

"""Very small historical simulation engine."""

import csv
import os
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

# Rule-based prediction knobs
SLOPE_THRESHOLD = 0.3       # slope > this = up
VOLATILITY_MAX = 24       # filter noisy/flat boxes
RANGE_MIN = 0.08           # must have enough range to matter

# Percent-change grading
STRONG_MOVE_THRESHOLD = 0.20  # 20% move is considered strong
COLOR_MAP = {
    -2: "darkred",
    -1: "pink",
    0: "gray",
    1: "lightgreen",
    2: "darkgreen",
}

FEATURES_CSV = "data/window_features.csv"


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


def rule_predict(features: dict[str, float]) -> int:
    """Classify next window move using percent-change categories."""
    if (
        features.get("volatility", 0.0) > VOLATILITY_MAX
        or features.get("range", 0.0) < RANGE_MIN
    ):
        return 0
    pct = features.get("pct_change", 0.0)
    if pct >= STRONG_MOVE_THRESHOLD:
        return 2
    if pct > 0:
        return 1
    if pct <= -STRONG_MOVE_THRESHOLD:
        return -2
    if pct < 0:
        return -1
    return 0


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

    bars: list[tuple[int, int, float]] = []
    markers: list[tuple[int, float, int]] = []  # (start_idx, price, prediction)
    last_features: dict[str, float] | None = None
    results: list[tuple[int, int]] = []  # (pred, actual)

    os.makedirs(os.path.dirname(FEATURES_CSV), exist_ok=True)

    for start in range(0, len(df) - WINDOW_SIZE, STEP_SIZE):
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

        pred: int | None = None
        if last_features is not None:
            pred = rule_predict(last_features)
            markers.append((start, level, pred))

        closes = sub["close"].values
        x = np.arange(len(closes))
        slope = float(np.polyfit(x, closes, 1)[0]) if len(closes) > 1 else 0.0
        volatility = float(np.std(closes)) if len(closes) else 0.0
        rng = high - low
        vol_mean = float(sub["volume"].mean()) if "volume" in sub else 0.0
        mid = len(sub) // 2
        early = float(sub["volume"].iloc[:mid].mean()) if mid and "volume" in sub else 0.0
        late = float(sub["volume"].iloc[mid:].mean()) if mid and "volume" in sub else 0.0
        volume_skew = ((late - early) / early) if early else 0.0
        exit_price = float(sub.iloc[-1]["close"])
        pct_change = (exit_price - level) / level if level else 0.0

        label: int
        if pct_change <= -STRONG_MOVE_THRESHOLD:
            label = -2
        elif pct_change < 0:
            label = -1
        elif pct_change >= STRONG_MOVE_THRESHOLD:
            label = 2
        elif pct_change > 0:
            label = 1
        else:
            label = 0

        actual = label
        if pred is not None:
            results.append((pred, actual))
            pred_str = {
                -2: "STRONG DOWN",
                -1: "MILD DOWN",
                0: "FLAT",
                1: "MILD UP",
                2: "STRONG UP",
            }[pred]
            act_str = {
                -2: "STRONG DOWN",
                -1: "MILD DOWN",
                0: "FLAT",
                1: "MILD UP",
                2: "STRONG UP",
            }[actual]
            print(f"start={start} predicted {pred_str} actual {act_str}")

        features = {
            "slope": slope,
            "volatility": volatility,
            "range": rng,
            "volume_mean": vol_mean,
            "volume_skew": volume_skew,
            "pct_change": pct_change,
            "label": label,
        }
        file_exists = os.path.exists(FEATURES_CSV)
        with open(FEATURES_CSV, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=features.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(features)

        last_features = {k: v for k, v in features.items() if k != "label"}

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

    for idx, price, pred in markers:
        if pred > 0:
            marker = "^"
        elif pred < 0:
            marker = "v"
        else:
            marker = "o"
        color = COLOR_MAP.get(pred, "gray")
        ax1.scatter(idx, price, marker=marker, color=color, zorder=5)

    ax1.set_title("Rolling Window Box Visualization")
    ax1.set_xlabel("Candles (Index)")
    ax1.set_ylabel("Price")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    made = sum(1 for p, _ in results if p != 0)
    correct = sum(1 for p, a in results if p != 0 and p == a)
    acc = 100 * correct / made if made else 0.0
    weights = {0: 1, 1: 1, -1: 1, 2: 2, -2: 2}
    total_w = sum(weights.get(a, 1) for p, a in results if p != 0)
    correct_w = sum(weights.get(a, 1) for p, a in results if p != 0 and p == a)
    w_acc = 100 * correct_w / total_w if total_w else 0.0
    print(
        f"Predictions made: {made} Correct: {correct} Accuracy: {acc:.2f}% Weighted: {w_acc:.2f}%"
    )

    plt.show()

