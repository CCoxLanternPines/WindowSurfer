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
SELL_TRIGGER = 10

FLAT_SELL_FRACTION = float(CONFIG.get("flat_sell_fraction", 0.2))
FLAT_SELL_THRESHOLD = float(CONFIG.get("flat_sell_threshold", 0.5))

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

    for t in range(WINDOW_SIZE - 1, len(df), WINDOW_STEP):
        start_idx = t - WINDOW_SIZE + 1
        end_idx = t + 1
        sub = df.iloc[start_idx:end_idx]

        # --- Extract features from backward-looking window ---
        low = float(sub["low"].min()) if "low" in sub else float(sub["close"].min())
        high = float(sub["high"].max()) if "high" in sub else float(sub["close"].max())
        level = float(df.iloc[start_idx]["close"])
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
        
        # Strictly forward outcome
        future_window = df.iloc[t : t + WINDOW_SIZE]  # candles t ... t+W-1
        if len(future_window) == WINDOW_SIZE:
            exit_price = float(future_window.iloc[-1]["close"])
            start_price = float(df.iloc[t]["close"])
            pct_change = (exit_price - start_price) / start_price
        else:
            pct_change = 0.0  # not enough future data at end of sim


        # --- Label actual move for accuracy stats ---
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

        # --- Multi-window crowd vote (replaces slope-flip) ---
        decision, confidence = multi_window_vote(
            df, t,
            window_sizes=[8, 12, 24, 48],
        )
        
        # --- Reversal detection using extreme disagreement ---
        turn_decision, turn_conf = multi_window_turnvote(
            df, t,
            window_sizes=[8, 16, 24]
        )

        if turn_decision == 1:  # bottom
            if viz:
                ax1.scatter(candle["candle_index"], candle["close"], color="lime", s=200, marker="^", zorder=7)
            print(f"[BOTTOM?] candle={candle['candle_index']} price={candle['close']:.2f} conf={turn_conf:.4f}")

        elif turn_decision == -1:  # top
            if viz:
                ax1.scatter(candle["candle_index"], candle["close"], color="magenta", s=200, marker="v", zorder=7)
            print(f"[TOP?] candle={candle['candle_index']} price={candle['close']:.2f} conf={turn_conf:.4f}")


        candle = df.iloc[t]
        if decision == 1:
            if viz:
                ax1.scatter(candle["candle_index"], candle["close"], color="green", s=120, zorder=6)
            print(f"[BUY]  candle={candle['candle_index']} price={candle['close']:.2f} confidence={confidence:.4f}")

        elif decision == -1:
            if viz:
                ax1.scatter(candle["candle_index"], candle["close"], color="red", s=120, zorder=6)
            print(f"[SELL] candle={candle['candle_index']} price={candle['close']:.2f} confidence={confidence:.4f}")

        # --- Record decision vs actual for stats ---
        if decision != 0:  # only log when we actually made a call
            results.append((decision, label, confidence))
            pred_str = { -1:"SELL", 0:"HOLD", 1:"BUY" }.get(decision, str(decision))
            act_str  = { -2:"STRONG DOWN", -1:"MILD DOWN", 0:"FLAT", 1:"MILD UP", 2:"STRONG UP" }[label]
            print(f"t={t} predicted {pred_str} actual {act_str}")

        # --- Write features to CSV for later analysis ---
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
        ax1.scatter([], [], color="orange", s=90, label="Flat Sell")

        ax1.set_title("Price with Trades")
        ax1.set_xlabel("Candles (Index)")
        ax1.set_ylabel("Price")
        ax1.legend(loc="upper left")
        ax1.grid(True)

    # --- Evaluate accuracy ---
    made = sum(1 for p, l, c in results if p != 0)
    correct = sum(1 for p, l, c in results if (p > 0 and l > 0) or (p < 0 and l < 0))
    total = len(results)
    accuracy = correct / total if total else 0

    # weighted accuracy using confidence
    weighted_correct = sum(c for p, l, c in results if (p > 0 and l > 0) or (p < 0 and l < 0))
    weighted_total   = sum(c for _, _, c in results)
    weighted_accuracy = weighted_correct / weighted_total if weighted_total else 0

    print(f"Predictions made: {made} Correct: {correct} "
      f"Accuracy: {accuracy*100:.2f}% Weighted: {weighted_accuracy*100:.2f}%")


    print(
        f"Skipped: slope={SLOPE_SKIPS} volatility={VOLATILITY_SKIPS} range={RANGE_SKIPS} skew_hits={SKEW_HITS}"
    )

    print(f"[RESULT] PnL={realized_pnl:.2f}, Remaining Notes={len(open_notes)}")

    if viz:
        plt.show()

def multi_window_vote(df, t, window_sizes, slope_thresh=0.001, range_thresh=0.05):
    votes = []
    strengths = []

    for W in window_sizes:
        if t - W < 0:
            continue
        sub = df.iloc[t - W : t]
        closes = sub["close"].values
        x = np.arange(len(closes))
        slope = float(np.polyfit(x, closes, 1)[0]) if len(closes) > 1 else 0.0
        rng = float(sub["close"].max() - sub["close"].min())

        # --- filtering ---
        if abs(slope) < slope_thresh:
            continue
        if rng < range_thresh:
            continue

        # --- vote ---
        direction = 1 if slope > 0 else -1
        votes.append(direction)

        # --- confidence contribution ---
        strength = abs(slope) * rng
        strengths.append(strength)

    score = sum(votes)
    confidence = sum(strengths) / max(1, len(strengths))  # average strength

    if score >= 2:
        return 1, confidence   # BUY
    elif score <= -2:
        return -1, confidence  # SELL
    else:
        return 0, confidence   # HOLD

def multi_window_turnvote(df, t, window_sizes, slope_thresh=0.001, range_thresh=0.05):
    slopes = {}
    strengths = {}

    for W in window_sizes:
        if t - W < 0: 
            continue
        sub = df.iloc[t-W:t]
        closes = sub["close"].values
        x = np.arange(len(closes))
        slope = float(np.polyfit(x, closes, 1)[0]) if len(closes) > 1 else 0.0
        rng = float(sub["close"].max() - sub["close"].min())

        if abs(slope) < slope_thresh or rng < range_thresh:
            continue

        slopes[W] = np.sign(slope)
        strengths[W] = abs(slope) * rng

    if not slopes:
        return 0, 0.0

    shortest = min(slopes.keys())
    longest  = max(slopes.keys())

    short_dir = slopes[shortest]
    long_dir  = slopes[longest]

    # Detect disagreement at extremes
    if short_dir > 0 and long_dir < 0:
        return 1, strengths[shortest]   # bottom candidate
    elif short_dir < 0 and long_dir > 0:
        return -1, strengths[shortest]  # top candidate
    else:
        return 0, np.mean(list(strengths.values()))



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", type=str, default="1m")
    parser.add_argument("--viz", action="store_true", help="Enable visualization")
    args = parser.parse_args()

    run_simulation(timeframe=args.time, viz=args.viz)


if __name__ == "__main__":
    main()

