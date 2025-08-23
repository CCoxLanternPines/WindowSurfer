from __future__ import annotations

"""Reversal brain identifying symmetry/overshoot patterns."""

from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..sim_engine import (
    WINDOW_SIZE,
    WINDOW_STEP,
    multi_window_vote,
)

LOOKBACK = 24
TOLERANCE = 0.10  # 10%


def classify(delta_before: float, delta_after: float) -> str:
    """Classify reversal symmetry or overshoot."""
    if delta_before == 0:
        return "fail"
    # Opposite sign indicates a potential reversal
    if delta_before * delta_after < 0:
        ratio = abs(delta_after) / abs(delta_before)
        if abs(1 - ratio) <= TOLERANCE:
            return "symmetry"
        if ratio > 1:
            return "overshoot"
    return "fail"


def run(df: pd.DataFrame, viz: bool):
    """Return reversal signals, optionally plotting them."""
    signals: List[Dict[str, float]] = []
    last_decision: int | None = None
    xs, ys, cats = [], [], []

    for t in range(WINDOW_SIZE - 1, len(df), WINDOW_STEP):
        decision, _, _ = multi_window_vote(df, t, window_sizes=[8, 12, 24, 48])
        if (
            last_decision is not None
            and decision != 0
            and decision != last_decision
            and t - LOOKBACK >= 0
            and t + LOOKBACK < len(df)
        ):
            x = int(df["candle_index"].iloc[t])
            y = float(df["close"].iloc[t])
            p_before = float(df["close"].iloc[t - LOOKBACK])
            p_after = float(df["close"].iloc[t + LOOKBACK])
            delta_before = y - p_before
            delta_after = p_after - y
            category = classify(delta_before, delta_after)
            signals.append(
                {
                    "index": x,
                    "price": y,
                    "category": category,
                    "delta_before": delta_before,
                    "delta_after": delta_after,
                }
            )
            print(
                f"[REV] idx={x} P_before={p_before:.2f} P_rev={y:.2f} "
                f"P_after={p_after:.2f} Δ_before={delta_before:.2f} Δ_after={delta_after:.2f} => {category}"
            )
            xs.append(x)
            ys.append(y)
            cats.append(category)
        if decision != 0:
            last_decision = decision

    if viz:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df["candle_index"], df["close"], lw=1, label="Close Price", color="blue")
        for x, y, cat in zip(xs, ys, cats):
            ax.scatter(x, y, color="yellow", edgecolors="black", zorder=6)
            ax.text(x, y, cat[0].upper(), color="black", fontsize=8, ha="center", va="center")
        ax.set_title("Price with Reversals (Brain)")
        ax.set_xlabel("Candles (Index)")
        ax.set_ylabel("Price")
        ax.grid(True)
        plt.show()

    return signals


def summarize(signals: List[Dict[str, float]], df: pd.DataFrame):
    """Aggregate reversal classification statistics."""
    total = len(signals)
    sym = sum(1 for s in signals if s["category"] == "symmetry")
    over = sum(1 for s in signals if s["category"] == "overshoot")
    fail = total - sym - over
    sym_pct = int(round(100 * sym / total)) if total else 0
    over_pct = int(round(100 * over / total)) if total else 0
    fail_pct = int(round(100 * fail / total)) if total else 0

    indices = [int(s["index"]) for s in signals]
    avg_gap = int(np.mean(np.diff(indices))) if len(indices) > 1 else 0
    median_gap = int(np.median(np.diff(indices))) if len(indices) > 1 else 0

    # --------------------------------------------------------------
    # Flip-to-extrema accuracy within ±5 candles
    # --------------------------------------------------------------
    highs = (
        df["close"].rolling(window=11, center=True).max().eq(df["close"])
    )
    lows = (
        df["close"].rolling(window=11, center=True).min().eq(df["close"])
    )
    extrema_hits = 0
    for s in signals:
        idx = int(s["index"])
        if s["delta_after"] < 0:  # flip down -> near local high
            if highs.iloc[idx - 5 : idx + 6].any():
                extrema_hits += 1
        elif s["delta_after"] > 0:  # flip up -> near local low
            if lows.iloc[idx - 5 : idx + 6].any():
                extrema_hits += 1
    flip_extrema_pct = int(round(100 * extrema_hits / total)) if total else 0

    # --------------------------------------------------------------
    # Follow-through strength Δ_after >= Δ_before
    # --------------------------------------------------------------
    followthrough = sum(
        1 for s in signals if abs(s["delta_after"]) >= abs(s["delta_before"])
    )
    followthrough_pct = int(round(100 * followthrough / total)) if total else 0

    # --------------------------------------------------------------
    # Next-window slope agreement (next 12 candles)
    # --------------------------------------------------------------
    slope_agree = 0
    closes = df["close"].to_numpy()
    for s in signals:
        idx = int(s["index"])
        y = closes[idx + 1 : idx + 13]
        if len(y) < 2:
            continue
        x_vals = np.arange(len(y))
        slope = float(np.polyfit(x_vals, y, 1)[0])
        if slope * s["delta_after"] > 0:
            slope_agree += 1
    slope_agree_pct = int(round(100 * slope_agree / total)) if total else 0

    print("[REV][stats]")
    print("  Symmetric reversals: {}%".format(sym_pct))
    print("  Overshoots: {}%".format(over_pct))
    print("  Fails: {}%".format(fail_pct))
    print("  Flip-extrema accuracy: {}%".format(flip_extrema_pct))
    print("  Follow-through >= before: {}%".format(followthrough_pct))
    print("  Next-slope agreement: {}%".format(slope_agree_pct))
    print("  Median gap: {} candles".format(median_gap))

    return {
        "count": total,
        "avg_gap": avg_gap,
        "slope_bias": f"{slope_agree_pct}%",
        "flip_extrema_pct": flip_extrema_pct,
        "followthrough_pct": followthrough_pct,
        "slope_agree_pct": slope_agree_pct,
        "median_gap": median_gap,
    }
