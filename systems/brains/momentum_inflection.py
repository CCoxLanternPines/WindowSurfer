from __future__ import annotations

"""
Brain 3 — Momentum Inflection (teal ▲)

What it does:
Detects slope weakening or bending back across mid-sized windows (24/48 bars)
to warn of impending momentum loss.

Strengths:
- Good "heads-up" that momentum is tiring.
- Often fires before pivots, giving early signals.

Weaknesses:
- Noisy in chop, many false alarms.
- Best when confirmed by Brain 1 (Exhaustion) or Brain 2 (Flip).

Key Questions:
- How quickly does momentum decay after the inflection?
- How early does this brain fire relative to Brain 2 flips?
- Does the move resolve as a continuation or a true reversal within 24 candles?
- Are signals sensitive to volatility regime (high vs low ATR)?
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
        ax.scatter(xs, ys, c="teal", marker="^", s=60, zorder=6)
        ax.set_title("Price with Momentum Inflection (Brain)")
        ax.set_xlabel("Candles (Index)")
        ax.set_ylabel("Price")
        ax.grid(True)
        plt.show()

    return signals


def summarize(signals: List[Dict[str, float]], df: pd.DataFrame):
    """Compute momentum-specific statistics for Brain 3."""
    count = len(signals)
    indices = [int(s["index"]) for s in signals]
    avg_gap = int(np.mean(np.diff(indices))) if len(indices) > 1 else 0
    up = sum(1 for s in signals if s["direction"] == "up")
    down = count - up
    pct = int(round(100 * max(up, down) / count)) if count else 0
    slope_bias = f"{'uptrend' if up >= down else 'downtrend'} {pct}%"

    closes = df["close"].to_numpy()

    # ------------------------------
    # Momentum Decay Ratio
    # ------------------------------
    ratios = [abs(s["slope_now"]) / abs(s["slope_prev"]) for s in signals if s["slope_prev"] != 0]
    slowdown_ratio_avg = float(np.mean(ratios)) if ratios else 0.0
    slowdown_collapse_pct = (
        int(round(100 * sum(r < 0.3 for r in ratios) / len(ratios))) if ratios else 0
    )

    # ------------------------------
    # Lead time vs Brain 2 (reversal)
    # ------------------------------
    from . import reversal
    import io, contextlib

    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
        rev_signals = reversal.run(df, viz=False)

    rev_indices = sorted(int(s["index"]) for s in rev_signals)
    lead_times: List[float] = []
    if rev_indices:
        rev_arr = np.array(rev_indices)
        for idx in indices:
            pos = np.searchsorted(rev_arr, idx)
            diffs: List[int] = []
            if pos < len(rev_arr):
                diffs.append(rev_arr[pos] - idx)
            if pos > 0:
                diffs.append(rev_arr[pos - 1] - idx)
            if diffs:
                lead_times.append(min(diffs, key=abs))

    leadtime_mean = float(np.mean(lead_times)) if lead_times else 0.0
    leadtime_median = float(np.median(lead_times)) if lead_times else 0.0

    # ------------------------------
    # Continuation vs Reversal Resolution
    # ------------------------------
    continuations = reversals = valid = 0
    for s in signals:
        idx = int(s["index"])
        slope_prev = float(s["slope_prev"])
        future = closes[idx + 1 : idx + 1 + LOOKAHEAD]
        if len(future) == LOOKAHEAD:
            slope_future = float(np.polyfit(np.arange(len(future)), future, 1)[0])
            valid += 1
            if slope_prev > 0:
                if slope_future > 0:
                    continuations += 1
                else:
                    reversals += 1
            else:
                if slope_future < 0:
                    continuations += 1
                else:
                    reversals += 1

    continuation_pct = int(round(100 * continuations / valid)) if valid else 0
    reversal_pct = 100 - continuation_pct if valid else 0

    # ------------------------------
    # Volatility Context Sensitivity (ATR14)
    # ------------------------------
    have_hl = all(col in df.columns for col in ["high", "low", "close"])
    if have_hl:
        prev_close = df["close"].shift(1)
        tr = pd.concat(
            [
                (df["high"] - df["low"]).abs(),
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(14, min_periods=1).mean()
    else:
        atr = df["close"].rolling(14, min_periods=1).std()

    signal_atr = atr.iloc[indices] if indices else pd.Series(dtype=float)
    median_atr = float(atr.median()) if len(atr) else 0.0
    high_vol_pct = (
        int(round(100 * (signal_atr > median_atr).sum() / len(signal_atr)))
        if len(signal_atr)
        else 0
    )
    low_vol_pct = 100 - high_vol_pct if len(signal_atr) else 0

    print("[BRAIN][momentum_inflection][stats]")
    print(
        f"  Avg slowdown ratio: {slowdown_ratio_avg:.2f} (collapse <0.3 = {slowdown_collapse_pct}%)"
    )
    print(
        f"  Lead time to flips: mean={leadtime_mean:.1f}c median={leadtime_median:.0f}c"
    )
    print(
        f"  Resolution: {reversal_pct}% reversals, {continuation_pct}% continuations"
    )
    print(
        f"  Volatility context: {high_vol_pct}% high-vol, {low_vol_pct}% low-vol"
    )

    return {
        "count": count,
        "avg_gap": avg_gap,
        "slope_bias": slope_bias,
        "slowdown_ratio_avg": slowdown_ratio_avg,
        "slowdown_collapse_pct": slowdown_collapse_pct,
        "leadtime_mean": leadtime_mean,
        "leadtime_median": leadtime_median,
        "continuation_pct": continuation_pct,
        "reversal_pct": reversal_pct,
        "high_vol_pct": high_vol_pct,
        "low_vol_pct": low_vol_pct,
    }
