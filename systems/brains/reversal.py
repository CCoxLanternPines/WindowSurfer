from __future__ import annotations

"""Reversal brain identifying symmetry/overshoot patterns."""

from typing import List, Dict

import matplotlib.pyplot as plt
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

    print("  Symmetric reversals: {}%".format(sym_pct))
    print("  Overshoots: {}%".format(over_pct))
    print("  Fails: {}%".format(fail_pct))

    return {"count": total, "avg_gap": 0, "slope_bias": 0}
