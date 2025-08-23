from __future__ import annotations

"""Brain 6 – Rolling peak detector with scorecard statistics."""

from typing import Dict, List

import contextlib
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WINDOW = 11


def run(df: pd.DataFrame, viz: bool) -> List[Dict[str, float]]:
    """Detect swing-high peaks (red stars)."""
    signals: List[Dict[str, float]] = []
    closes = df["close"].to_numpy()
    xs, ys = [], []

    for t in range(WINDOW, len(df)):
        window = closes[t - WINDOW : t + 1]
        if closes[t] == window.max():
            idx = int(df["candle_index"].iloc[t])
            price = float(closes[t])
            signals.append({"index": idx, "price": price})
            xs.append(idx)
            ys.append(price)

    if viz:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df["candle_index"], df["close"], lw=1, color="blue")
        ax.scatter(xs, ys, c="red", marker="*", s=120, zorder=6)
        ax.set_title("Price with Rolling Peaks (Brain 6)")
        ax.set_xlabel("Candles (Index)")
        ax.set_ylabel("Price")
        ax.grid(True)
        plt.show()

    return signals


def summarize(signals: List[Dict[str, float]], df: pd.DataFrame):
    """Aggregate rolling-peak statistics."""
    total = len(signals)
    indices = [int(s["index"]) for s in signals]
    prices = [float(s["price"]) for s in signals]
    transitions = max(0, total - 1)

    gaps = np.diff(indices) if total > 1 else np.array([])
    avg_gap = int(np.mean(gaps)) if len(gaps) else 0

    continuation = 0
    reversal = 0
    drawdowns: List[float] = []

    for i in range(transitions):
        cur_idx, next_idx = indices[i], indices[i + 1]
        cur_price, next_price = prices[i], prices[i + 1]
        if next_price >= cur_price:
            continuation += 1
        else:
            reversal += 1
            trough = float(df["close"].iloc[cur_idx:next_idx].min())
            drawdowns.append((trough - cur_price) / cur_price * 100)

    continuation_pct = (
        int(round(100 * continuation / transitions)) if transitions else 0
    )
    reversal_pct = int(round(100 * reversal / transitions)) if transitions else 0
    avg_drawdown = float(np.mean(drawdowns)) if drawdowns else 0.0

    gap_avg = float(np.mean(gaps)) if len(gaps) else 0.0
    gap_tight_pct = int(round(100 * np.sum(gaps < 10) / len(gaps))) if len(gaps) else 0
    gap_wide_pct = int(round(100 * np.sum(gaps > 30) / len(gaps))) if len(gaps) else 0

    cluster_dips: List[float] = []
    if transitions:
        lows = df["low"] if "low" in df.columns else df["close"]
        for i in range(transitions):
            gap = indices[i + 1] - indices[i]
            if gap <= 10:
                star_price = prices[i]
                between = lows.iloc[indices[i] + 1 : indices[i + 1]]
                trough = float(between.min()) if not between.empty else star_price
                cluster_dips.append((trough / star_price - 1) * 100)

    up = sum(1 for i in range(transitions) if prices[i + 1] >= prices[i])
    down = transitions - up
    slope_bias = (
        f"{'uptrend' if up >= down else 'downtrend'} {int(round(100 * max(up, down) / transitions))}%"
        if transitions
        else "flat 0%"
    )

    avg_cluster_dip = float(np.mean(cluster_dips)) if cluster_dips else 0.0
    median_cluster_dip = float(np.median(cluster_dips)) if cluster_dips else 0.0
    gt2pct_dip_pct = (
        int(round(100 * np.sum(np.array(cluster_dips) <= -2) / len(cluster_dips)))
        if cluster_dips
        else 0
    )

    overlap_div_pct = 0
    overlap_combo_pct = 0
    if total:
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            from . import divergence, reversal as rev

            div_signals = divergence.run(df, viz=False)
            rev_signals = rev.run(df, viz=False)
            combo_indices = divergence._brain9_combo(
                [int(s["index"]) for s in div_signals],
                [int(s["index"]) for s in rev_signals],
            )

        div_indices = [int(s["index"]) for s in div_signals]
        N = 2

        def overlap_count(src: List[int]) -> int:
            cnt = 0
            for idx in indices:
                for sidx in src:
                    if abs(idx - sidx) <= N:
                        cnt += 1
                        break
            return cnt

        overlap_div = overlap_count(div_indices)
        overlap_combo = overlap_count(combo_indices)
        overlap_div_pct = int(round(100 * overlap_div / total))
        overlap_combo_pct = int(round(100 * overlap_combo / total))

    print("[BRAIN][rolling_peak][stats]")
    print(f"  Continuation rate: {continuation_pct}%")
    print(
        f"  Reversal rate: {reversal_pct}% (avg drawdown {avg_drawdown:.1f}%)"
    )
    print(
        f"  Star gap avg: {gap_avg:.0f}c | tight (<10c): {gap_tight_pct}% | wide (>30c): {gap_wide_pct}%"
    )
    print(
        f"  Overlaps: Divergence {overlap_div_pct}% | Combo {overlap_combo_pct}%"
    )
    print(f"  Avg dip in tight clusters: {avg_cluster_dip:.1f}%")
    print(f"  Median dip: {median_cluster_dip:.1f}%")
    print(f"  >2% dips: {gt2pct_dip_pct}%")

    return {
        "count": total,
        "avg_gap": avg_gap,
        "slope_bias": slope_bias,
        "continuation_pct": continuation_pct,
        "reversal_pct": reversal_pct,
        "avg_drawdown": avg_drawdown,
        "gap_avg": gap_avg,
        "gap_tight_pct": gap_tight_pct,
        "gap_wide_pct": gap_wide_pct,
        "overlap_div_pct": overlap_div_pct,
        "overlap_combo_pct": overlap_combo_pct,
        "avg_cluster_dip": avg_cluster_dip,
        "median_cluster_dip": median_cluster_dip,
        "gt2pct_dip_pct": gt2pct_dip_pct,
    }
