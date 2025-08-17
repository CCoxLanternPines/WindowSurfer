from __future__ import annotations

"""Very small historical simulation engine."""

import argparse
import csv
import json
import os
import re
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from systems.scripts.math.slope_score import classify_slope


# Box / rolling window parameters
WINDOW_SIZE = 24  # candles per box
WINDOW_STEP = 2   # rolling step

# Percent-change grading
STRONG_MOVE_THRESHOLD = 0.15   # slightly easier "strong move"

# Prediction filters
RANGE_MIN = 0.08               # only 8%+ range windows matter
VOLUME_SKEW_BIAS = 0.4         # allow moderate skew to count

# Load optional configuration knobs
try:
    with open("settings/config.json", "r") as _cfg:
        CONFIG = json.load(_cfg)
except FileNotFoundError:  # pragma: no cover - optional config
    CONFIG = {}

FLAT_BAND_DEG = float(CONFIG.get("flat_band_deg", 10.0))

MAX_PRESSURE = 10.0
BUY_TRIGGER = 3.0
SELL_TRIGGER = 3.0

FEATURES_CSV = "data/window_features.csv"


# Debug counters for filter skips
SLOPE_SKIPS = 0
VOLATILITY_SKIPS = 0
RANGE_SKIPS = 0
SKEW_HITS = 0


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


def rule_predict(features: dict[str, float]) -> int:
    """Classify next window move with multi-feature rules."""
    global SLOPE_SKIPS, RANGE_SKIPS, SKEW_HITS

    slope = features.get("slope", 0.0)
    rng = features.get("range", 0.0)

    slope_cls = classify_slope(slope, FLAT_BAND_DEG)
    if slope_cls == 0:
        SLOPE_SKIPS += 1
        return 0
    if rng < RANGE_MIN:
        RANGE_SKIPS += 1
        return 0

    skew = features.get("volume_skew", 0.0)
    if skew > VOLUME_SKEW_BIAS and slope_cls > 0:
        SKEW_HITS += 1
        return 1
    if skew < -VOLUME_SKEW_BIAS and slope_cls < 0:
        SKEW_HITS += 1
        return -1

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


def run_simulation(*, timeframe: str = "1m", viz: bool = True) -> None:
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

    if viz:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(df["candle_index"], df["close"], color="blue", label="Close Price")
    else:
        ax1 = None

    last_features: dict[str, float] | None = None
    results: list[tuple[int, int]] = []  # (pred, actual)

    buy_pressure = 0.0
    sell_pressure = 0.0

    open_notes: list[tuple[float, float]] = []  # (price, size)
    realized_pnl = 0.0

    os.makedirs(os.path.dirname(FEATURES_CSV), exist_ok=True)

    for start in range(0, len(df) - WINDOW_SIZE, WINDOW_STEP):
        end = start + WINDOW_SIZE
        sub = df.iloc[start:end]

        low = float(sub["low"].min()) if "low" in sub else float(sub["close"].min())
        high = float(sub["high"].max()) if "high" in sub else float(sub["close"].max())
        level = float(df.iloc[start]["close"])

        pred: int | None = None
        if last_features is not None:
            pred = rule_predict(last_features)

            if pred > 0:  # upward prediction
                buy_pressure = min(MAX_PRESSURE, buy_pressure + 1)
                sell_pressure = max(0.0, sell_pressure - 2)
            elif pred < 0:  # downward prediction
                sell_pressure = min(MAX_PRESSURE, sell_pressure + 1)
                buy_pressure = max(0.0, buy_pressure - 2)
            else:  # neutral
                buy_pressure = max(0.0, buy_pressure - 0.5)
                sell_pressure = max(0.0, sell_pressure - 0.5)

            candle = df.iloc[start]
            print(
                f"[PRESSURE] candle={candle['candle_index']} buy={buy_pressure:.1f} sell={sell_pressure:.1f}"
            )

            if buy_pressure >= BUY_TRIGGER:
                trade_size = buy_pressure / MAX_PRESSURE
                open_notes.append((candle["close"], trade_size))
                if viz:
                    ax1.scatter(
                        candle["candle_index"],
                        candle["close"],
                        color="green",
                        s=120,
                        zorder=6,
                    )
                print(
                    f"[BUY] candle={candle['candle_index']} price={candle['close']} size={trade_size:.2f}"
                )
                buy_pressure = 0.0

            if sell_pressure >= SELL_TRIGGER and open_notes:
                total_size = sum(size for _, size in open_notes)
                sell_frac = sell_pressure / MAX_PRESSURE
                trade_size = sell_frac * total_size
                remaining = trade_size
                pnl = 0.0
                while remaining > 1e-9 and open_notes:
                    entry_price, entry_size = open_notes[0]
                    if entry_size <= remaining:
                        pnl += (candle["close"] - entry_price) * entry_size
                        remaining -= entry_size
                        open_notes.pop(0)
                    else:
                        pnl += (candle["close"] - entry_price) * remaining
                        open_notes[0] = (entry_price, entry_size - remaining)
                        remaining = 0.0
                realized_pnl += pnl
                if viz:
                    ax1.scatter(
                        candle["candle_index"],
                        candle["close"],
                        color="red",
                        s=120,
                        zorder=6,
                    )
                print(
                    f"[SELL] candle={candle['candle_index']} price={candle['close']} size={trade_size:.2f} pnl={pnl:.2f} total={realized_pnl:.2f}"
                )
                sell_pressure = 0.0

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

    if viz:
        ax1.scatter([], [], color="green", s=120, label="Buy")
        ax1.scatter([], [], color="red", s=120, label="Sell")

        ax1.set_title("Price with Trades")
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
    print(
        f"Skipped: slope={SLOPE_SKIPS} volatility={VOLATILITY_SKIPS} range={RANGE_SKIPS} skew_hits={SKEW_HITS}"
    )

    print(f"[RESULT] PnL={realized_pnl:.2f}, Remaining Notes={len(open_notes)}")

    if viz:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", type=str, default="1m")
    parser.add_argument("--viz", action="store_true", help="Enable visualization")
    args = parser.parse_args()

    run_simulation(timeframe=args.time, viz=args.viz)


if __name__ == "__main__":
    main()

