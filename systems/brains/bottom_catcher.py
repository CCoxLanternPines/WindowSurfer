from __future__ import annotations

"""Brain 4: Bottom catcher with bounce statistics."""

from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..sim_engine import WINDOW_SIZE, WINDOW_STEP


X_THRESH = 0.02  # 2% move
LOOKAHEAD = 24   # candles
ALIGN_WINDOW = 5  # ±5 candles for extrema alignment


def run(df: pd.DataFrame, viz: bool):
    """Detect local bottoms with improving slope."""
    signals: List[Dict[str, float]] = []
    xs, ys = [], []

    slope_now = 0.0
    slope_prev = 0.0

    for t in range(WINDOW_SIZE - 1, len(df), WINDOW_STEP):
        if t >= 48:
            sub_now = df["close"].iloc[t-24:t]
            sub_prev = df["close"].iloc[t-48:t-24]
            slope_now = float(np.polyfit(np.arange(len(sub_now)), sub_now, 1)[0]) if len(sub_now) > 1 else 0.0
            slope_prev = float(np.polyfit(np.arange(len(sub_prev)), sub_prev, 1)[0]) if len(sub_prev) > 1 else 0.0
        if t >= 36:
            lookback = 12
            window = df["close"].iloc[t-lookback:t+1]
            if df["close"].iloc[t] == float(window.min()) and slope_now > slope_prev:
                x = int(df["candle_index"].iloc[t])
                y = float(df["close"].iloc[t])
                signals.append({"index": x, "price": y})
                xs.append(x)
                ys.append(y)

    if viz:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df["candle_index"], df["close"], lw=1, label="Close Price", color="blue")
        ax.scatter(xs, ys, color="cyan", marker="v", s=100, zorder=6)
        ax.set_title("Price with Bottom Catcher (Brain 4)")
        ax.set_xlabel("Candles (Index)")
        ax.set_ylabel("Price")
        ax.grid(True)
        plt.show()

    return signals


def summarize(signals: List[Dict[str, float]], df: pd.DataFrame):
    """Compute bounce and continuation statistics."""
    total = len(signals)
    indices = [int(s["index"]) for s in signals]
    avg_gap = int(np.mean(np.diff(indices))) if len(indices) > 1 else 0

    closes = df["close"].to_numpy()
    success = 0
    bounce_mags: List[float] = []
    align_hits = 0
    continuation = 0
    false_bottom = 0

    for idx in indices:
        if idx >= len(closes):
            continue
        entry = closes[idx]
        future = closes[idx + 1 : idx + 1 + LOOKAHEAD]
        if len(future) == 0:
            continue
        max_future = float(future.max())
        min_future = float(future.min())
        bounce_mag = (max_future - entry) / entry
        bounce_mags.append(bounce_mag)
        if max_future >= entry * (1 + X_THRESH):
            success += 1
        if min_future <= entry * (1 - X_THRESH):
            false_bottom += 1
        low_window = closes[max(0, idx - ALIGN_WINDOW) : idx + ALIGN_WINDOW + 1]
        if entry == float(low_window.min()):
            align_hits += 1
        cont_window = closes[idx + 12 : idx + 24]
        if len(cont_window) >= 2:
            x_vals = np.arange(len(cont_window))
            slope = float(np.polyfit(x_vals, cont_window, 1)[0])
            if slope > 0:
                continuation += 1

    bounce_success_pct = int(round(100 * success / total)) if total else 0
    avg_bounce_pct = 100 * (np.mean(bounce_mags) if bounce_mags else 0.0)
    extrema_align_pct = int(round(100 * align_hits / total)) if total else 0
    continuation_pct = int(round(100 * continuation / total)) if total else 0
    false_bottom_pct = int(round(100 * false_bottom / total)) if total else 0

    print("[BRAIN][bottom_catcher][stats]")
    print(f"  Bounce success rate: {bounce_success_pct}%")
    print(f"  Avg bounce magnitude: {avg_bounce_pct:+.1f}%")
    print(f"  Extrema alignment: {extrema_align_pct}%")
    print(f"  Direction continuation: {continuation_pct}%")
    print(f"  False bottoms: {false_bottom_pct}%")

    return {
        "count": total,
        "avg_gap": avg_gap,
        "slope_bias": f"{continuation_pct}%",
        "bounce_success_pct": bounce_success_pct,
        "avg_bounce_pct": round(avg_bounce_pct, 2),
        "extrema_align_pct": extrema_align_pct,
        "continuation_pct": continuation_pct,
        "false_bottom_pct": false_bottom_pct,
    }
