from __future__ import annotations

"""
Brain 3 — Momentum Inflection (teal squares)

What it does:
Detects slope weakening or bending back across mid-sized windows (24/48 bars).
Purpose is to act as an early warning before a full reversal.

Strengths:
- Good “heads-up” that momentum is tiring.
- Often fires before pivots, giving early signals.

Weaknesses:
- Noisy in chop, many false alarms.
- Best when confirmed by Brain 1 (Exhaustion) or Brain 2 (Flip).

Key Questions:
- Is this slowdown followed by an actual reversal soon?
- How strong is the bounce that follows?
- How often does it fail and trend continues?
"""

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WINDOW = 24
LONG_WINDOW = 48
LOOKAHEAD = 24


def run(df: pd.DataFrame, viz: bool):
    """Return momentum inflection signals, optionally plotting them."""
    signals: List[Dict[str, float]] = []
    xs, ys = [], []
    closes = df["close"].to_numpy()
    xvals = np.arange(WINDOW)

    for t in range(LONG_WINDOW, len(df)):
        if t - WINDOW - 1 < 0:
            continue
        y_now = closes[t - WINDOW + 1 : t + 1]
        y_prev = closes[t - WINDOW : t]
        slope_now = float(np.polyfit(xvals, y_now, 1)[0])
        slope_prev = float(np.polyfit(xvals, y_prev, 1)[0])
        if slope_prev == 0:
            continue
        if slope_prev * slope_now < 0 or abs(slope_now) < abs(slope_prev) * 0.5:
            idx = int(df["candle_index"].iloc[t])
            price = float(closes[t])
            direction = "up" if slope_prev > 0 else "down"
            signals.append(
                {
                    "index": idx,
                    "price": price,
                    "slope_now": slope_now,
                    "slope_prev": slope_prev,
                    "direction": direction,
                }
            )
            xs.append(idx)
            ys.append(price)

    if viz:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df["candle_index"], df["close"], lw=1, color="blue")
        ax.scatter(xs, ys, c="teal", marker="s", s=60, zorder=6)
        ax.set_title("Price with Momentum Inflection (Brain)")
        ax.set_xlabel("Candles (Index)")
        ax.set_ylabel("Price")
        ax.grid(True)
        plt.show()

    return signals


def summarize(signals: List[Dict[str, float]], df: pd.DataFrame):
    """Compute momentum inflection statistics."""
    count = len(signals)
    indices = [int(s["index"]) for s in signals]
    avg_gap = int(np.mean(np.diff(indices))) if len(indices) > 1 else 0
    up = sum(1 for s in signals if s["direction"] == "up")
    down = count - up
    pct = int(round(100 * max(up, down) / count)) if count else 0
    slope_bias = f"{'uptrend' if up >= down else 'downtrend'} {pct}%"

    closes = df["close"].to_numpy()

    ratios: List[float] = []
    rev_flags: List[int] = []
    dists: List[float] = []
    bounce_moves: List[float] = []
    reversals = valid_rev = 0
    persistence_fails = valid_persist = 0
    direct_rev = valid_direct = 0

    highs = df["close"].rolling(window=25, center=True).max().eq(df["close"])
    lows = df["close"].rolling(window=25, center=True).min().eq(df["close"])

    for s in signals:
        idx = int(s["index"])
        price = float(closes[idx])
        slope_now = float(s["slope_now"])
        slope_prev = float(s["slope_prev"])
        direction = s["direction"]

        if abs(slope_prev) > 1e-9:
            ratios.append(abs(slope_now) / abs(slope_prev))

        future = closes[idx + 1 : idx + 1 + LOOKAHEAD]
        if len(future) >= 2:
            slope_future = float(np.polyfit(np.arange(len(future)), future, 1)[0])
            rev = (direction == "up" and slope_future < 0) or (
                direction == "down" and slope_future > 0
            )
            rev_flags.append(1 if rev else 0)
            if rev:
                reversals += 1
            valid_rev += 1

        start = max(0, idx - 12)
        end = min(len(df) - 1, idx + 12)
        extrema = highs.iloc[start : end + 1] | lows.iloc[start : end + 1]
        if extrema.any():
            pos = np.where(extrema.to_numpy())[0]
            dists.append(float(np.min(np.abs(pos - (idx - start)))))

        future24 = closes[idx + 1 : idx + 25]
        if len(future24) == 24:
            valid_persist += 1
            if direction == "up":
                if future24.max() >= price * 1.02:
                    persistence_fails += 1
            else:
                if future24.min() <= price * 0.98:
                    persistence_fails += 1

        start_b = idx + 10
        end_b = min(idx + 24, len(closes) - 1)
        if start_b < len(closes):
            window_b = closes[start_b : end_b + 1]
            if len(window_b) > 0:
                if direction == "up":
                    move = (price - window_b.min()) / price * 100
                else:
                    move = (window_b.max() - price) / price * 100
                bounce_moves.append(move)

        future10 = closes[idx + 1 : idx + 11]
        if len(future10) == 10:
            valid_direct += 1
            if direction == "up" and np.all(future10 < price):
                direct_rev += 1
            elif direction == "down" and np.all(future10 > price):
                direct_rev += 1

    reversal_rate = int(round(100 * reversals / valid_rev)) if valid_rev else 0
    avg_slowdown = float(np.mean(ratios)) if ratios else 0.0
    slowdown_corr = (
        float(np.corrcoef(ratios, rev_flags)[0, 1])
        if ratios and len(set(rev_flags)) > 1
        else 0.0
    )
    extrema_dist = float(np.mean(dists)) if dists else 0.0
    persistence_fail_pct = (
        int(round(100 * persistence_fails / valid_persist)) if valid_persist else 0
    )
    bounce_pct = float(np.mean(bounce_moves)) if bounce_moves else 0.0
    direct_reversal_pct = (
        int(round(100 * direct_rev / valid_direct)) if valid_direct else 0
    )

    print("[BRAIN][momentum_inflection][stats]")
    print(f"  Reversal rate: {reversal_rate}%")
    print(f"  Avg slowdown ratio: {avg_slowdown:.2f} (corr={slowdown_corr:.2f})")
    print(f"  Extrema distance: {extrema_dist:.1f} candles")
    print(f"  False persistence: {persistence_fail_pct}%")
    print(f"  Avg bounce-back: {bounce_pct:+.1f}%")
    print(f"  Direct reversals (10c): {direct_reversal_pct}%")

    return {
        "count": count,
        "avg_gap": avg_gap,
        "slope_bias": slope_bias,
        "reversal_rate": reversal_rate,
        "slowdown_ratio": avg_slowdown,
        "extrema_dist": extrema_dist,
        "persistence_fail_pct": persistence_fail_pct,
        "bounce_pct": bounce_pct,
        "direct_reversal_pct": direct_reversal_pct,
    }
