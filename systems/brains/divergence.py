from __future__ import annotations

"""Brain 5 – Divergence detector with predictive scorecard."""

from typing import Dict, List

import contextlib
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WINDOW = 24
LOOKAHEAD = 24
FALSE_CONT_PCT = 0.02  # 2% continuation threshold


def run(df: pd.DataFrame, viz: bool):
    """Detect bearish divergences using slope weakening."""
    signals: List[Dict[str, float]] = []
    xs, ys = [], []
    closes = df["close"].to_numpy()
    xvals = np.arange(WINDOW)

    for t in range(WINDOW * 2, len(df)):
        price_now = closes[t - 1]
        price_prev = closes[t - 1 - WINDOW]
        sub_now = closes[t - WINDOW : t]
        sub_prev = closes[t - 2 * WINDOW : t - WINDOW]
        slope_now = float(np.polyfit(xvals, sub_now, 1)[0]) if len(sub_now) > 1 else 0.0
        slope_prev = (
            float(np.polyfit(xvals, sub_prev, 1)[0]) if len(sub_prev) > 1 else 0.0
        )
        if price_now > price_prev and slope_now < slope_prev:
            idx = int(df["candle_index"].iloc[t])
            price = float(closes[t])
            signals.append(
                {
                    "index": idx,
                    "price": price,
                    "slope_now": slope_now,
                    "slope_prev": slope_prev,
                }
            )
            xs.append(idx)
            ys.append(price)

    if viz:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df["candle_index"], df["close"], lw=1, color="blue")
        ax.scatter(xs, ys, c="orange", marker="s", s=110, zorder=6)
        ax.set_title("Price with Divergence (Brain 5)")
        ax.set_xlabel("Candles (Index)")
        ax.set_ylabel("Price")
        ax.grid(True)
        plt.show()

    return signals


def _brain6_peaks(df: pd.DataFrame) -> List[int]:
    """Simple swing-high detector (Brain 6)."""
    closes = df["close"].to_numpy()
    indices: List[int] = []
    for t in range(11, len(closes)):
        window = closes[t - 11 : t + 1]
        if closes[t] == window.max():
            indices.append(int(df["candle_index"].iloc[t]))
    return indices


def _brain9_combo(div_indices: List[int], rev_indices: List[int]) -> List[int]:
    """Combo-top where divergence and reversal align (Brain 9)."""
    combos: List[int] = []
    i = j = 0
    div_indices = sorted(div_indices)
    rev_indices = sorted(rev_indices)
    while i < len(div_indices) and j < len(rev_indices):
        d = div_indices[i]
        r = rev_indices[j]
        if abs(d - r) <= 2:
            combos.append(r)
            i += 1
            j += 1
        elif d < r:
            i += 1
        else:
            j += 1
    return combos


def summarize(signals: List[Dict[str, float]], df: pd.DataFrame):
    """Compute divergence scorecard statistics."""
    total = len(signals)
    indices = [int(s["index"]) for s in signals]
    avg_gap = int(np.mean(np.diff(indices))) if len(indices) > 1 else 0
    up = sum(1 for s in signals if s["slope_prev"] > 0)
    down = total - up
    slope_bias = (
        f"{'uptrend' if up >= down else 'downtrend'} {int(round(100 * max(up, down) / total))}%"
        if total
        else "flat 0%"
    )

    # Brain 6 (peaks) and Brain 9 (combo tops)
    peak_indices = _brain6_peaks(df)
    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
        from . import reversal

        rev_signals = reversal.run(df, viz=False)
    rev_indices = [int(s["index"]) for s in rev_signals]
    combo_indices = _brain9_combo(indices, rev_indices)
    top_indices = np.array(sorted(peak_indices + combo_indices))

    closes = df["close"].to_numpy()
    div_to_top = 0
    lead_times: List[int] = []
    false_cont = 0
    valid_false = 0
    deltas: List[float] = []
    top_flags: List[int] = []

    for s in signals:
        idx = int(s["index"])
        delta_slope = float(s["slope_prev"]) - float(s["slope_now"])
        deltas.append(delta_slope)

        pos = np.searchsorted(top_indices, idx)
        next_top = None
        if pos < len(top_indices):
            diff = int(top_indices[pos]) - idx
            if diff >= 0:
                next_top = int(top_indices[pos])

        if next_top is not None and next_top - idx <= LOOKAHEAD:
            div_to_top += 1
            lead_times.append(next_top - idx)
            top_flags.append(1)
        else:
            top_flags.append(0)

        if idx + LOOKAHEAD < len(closes):
            valid_false += 1
            if next_top is None or next_top - idx > LOOKAHEAD:
                future_max = float(closes[idx + 1 : idx + LOOKAHEAD + 1].max())
                if future_max >= closes[idx] * (1 + FALSE_CONT_PCT):
                    false_cont += 1

    div_to_top_pct = int(round(100 * div_to_top / total)) if total else 0
    leadtime_mean = float(np.mean(lead_times)) if lead_times else 0.0
    leadtime_median = float(np.median(lead_times)) if lead_times else 0.0
    false_cont_pct = int(round(100 * false_cont / valid_false)) if valid_false else 0

    strength_corr = 0.0
    if deltas and len(set(top_flags)) > 1 and len(set(deltas)) > 1:
        strength_corr = float(np.corrcoef(deltas, top_flags)[0, 1])

    print("[BRAIN][divergence][stats]")
    print(f"  Divergence → Top: {div_to_top_pct}%")
    print(
        f"  Lead time to peak: mean={leadtime_mean:.1f}c median={leadtime_median:.0f}c"
    )
    print(f"  False continuations: {false_cont_pct}%")
    print(f"  Strength correlation: {strength_corr:.2f}")

    return {
        "count": total,
        "avg_gap": avg_gap,
        "slope_bias": slope_bias,
        "div_to_top_pct": div_to_top_pct,
        "leadtime_mean": leadtime_mean,
        "leadtime_median": leadtime_median,
        "false_cont_pct": false_cont_pct,
        "strength_corr": strength_corr,
    }
