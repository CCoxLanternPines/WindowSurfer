from __future__ import annotations

"""Exhaustion brain producing red/green cluster signals."""

from collections import deque
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..utils.brain_math import (
    WINDOW_SIZE,
    WINDOW_STEP,
    CLUSTER_WINDOW,
    BASE_SIZE,
    SCALE_POWER,
    multi_window_vote,
)


def run(df: pd.DataFrame, viz: bool):
    """Return exhaustion signals, optionally plotting them."""
    signals: List[Dict[str, float]] = []
    recent_buys = deque(maxlen=CLUSTER_WINDOW)
    recent_sells = deque(maxlen=CLUSTER_WINDOW)

    xr, yr, sr = [], [], []
    xg, yg, sg = [], [], []

    for t in range(WINDOW_SIZE - 1, len(df), WINDOW_STEP):
        x = int(df["candle_index"].iloc[t])
        y = float(df["close"].iloc[t])
        decision, _, _ = multi_window_vote(df, t, window_sizes=[8, 12, 24, 48])
        if decision == 1:
            recent_buys.append(t)
            cluster_strength = sum(1 for idx in recent_buys if t - idx <= CLUSTER_WINDOW)
            size = BASE_SIZE * (cluster_strength ** SCALE_POWER)
            xr.append(x)
            yr.append(y)
            sr.append(size)
            signals.append({"candle_index": x, "price": y, "direction": "sell", "size": size})
        elif decision == -1:
            recent_sells.append(t)
            cluster_strength = sum(1 for idx in recent_sells if t - idx <= CLUSTER_WINDOW)
            size = BASE_SIZE * (cluster_strength ** SCALE_POWER)
            xg.append(x)
            yg.append(y)
            sg.append(size)
            signals.append({"candle_index": x, "price": y, "direction": "buy", "size": size})

    if viz:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df["candle_index"], df["close"], lw=1, label="Close Price", color="blue")
        ax.scatter(xr, yr, s=sr, c="red", zorder=6)
        ax.scatter(xg, yg, s=sg, c="green", zorder=6)
        ax.set_title("Price with Exhaustion (Brain)")
        ax.set_xlabel("Candles (Index)")
        ax.set_ylabel("Price")
        ax.grid(True)
        plt.show()

    return signals


def summarize(signals: List[Dict[str, float]], df: pd.DataFrame):
    """Compute exhaustion run and reversal statistics."""

    # ------------------------------------------------------------------
    # Basic stats maintained for compatibility with ``brain_engine``.
    # ------------------------------------------------------------------
    count = len(signals)
    indices = [s["candle_index"] for s in signals]
    avg_gap = int(np.mean(np.diff(indices))) if len(indices) > 1 else 0
    up = sum(1 for s in signals if s["direction"] == "sell")
    down = sum(1 for s in signals if s["direction"] == "buy")
    total = up + down
    if total:
        pct = int(round(100 * up / total))
        direction = "uptrend" if up >= down else "downtrend"
    else:
        pct = 0
        direction = "flat"

    # ------------------------------------------------------------------
    # Trend run tracking
    # ------------------------------------------------------------------
    runs: List[Dict[str, float]] = []
    current_dir: int | None = None
    start_idx: int | None = None
    max_pressure = 0.0

    for s in sorted(signals, key=lambda x: x["candle_index"]):
        idx = int(s["candle_index"])
        direction_val = 1 if s["direction"] == "buy" else -1
        size = float(s["size"])

        if current_dir is None:
            current_dir = direction_val
            start_idx = idx
            max_pressure = size
            continue

        if direction_val == current_dir:
            if size > max_pressure:
                max_pressure = size
            continue

        # Direction flipped – close out previous run
        if start_idx is not None:
            runs.append(
                {
                    "dir": current_dir,
                    "length": idx - start_idx,
                    "max_pressure": max_pressure,
                    "reversal_idx": idx,
                }
            )

        # Start new run
        current_dir = direction_val
        start_idx = idx
        max_pressure = size

    # ------------------------------------------------------------------
    # Aggregate statistics
    # ------------------------------------------------------------------
    up_runs = [r for r in runs if r["dir"] == 1]
    down_runs = [r for r in runs if r["dir"] == -1]

    def _mean(values: List[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    avg_uptrend_duration = _mean([r["length"] for r in up_runs if r["length"] > 12])
    avg_downtrend_duration = _mean([r["length"] for r in down_runs if r["length"] > 12])
    avg_uptrend_pressure = _mean([r["max_pressure"] for r in up_runs])
    avg_downtrend_pressure = _mean([r["max_pressure"] for r in down_runs])

    # ------------------------------------------------------------------
    # Reversal slope statistics (24 candles after flip)
    # ------------------------------------------------------------------
    up_slopes: List[float] = []
    down_slopes: List[float] = []
    closes = df["close"].tolist()
    for r in runs:
        idx = int(r["reversal_idx"])
        if idx + 24 <= len(closes):
            y = closes[idx : idx + 24]
            x = list(range(len(y)))
            slope = float(np.polyfit(x, y, 1)[0])
            if r["dir"] == 1:
                up_slopes.append(slope)
            else:
                down_slopes.append(slope)

    avg_up_reversal_slope24 = _mean(up_slopes)
    avg_down_reversal_slope24 = _mean(down_slopes)

    # ------------------------------------------------------------------
    # Additional high-value statistics
    # ------------------------------------------------------------------

    # Edge accuracy: exhaustion near local extrema within ±5 candles
    highs = df["close"].rolling(window=11, center=True).max().eq(df["close"])
    lows = df["close"].rolling(window=11, center=True).min().eq(df["close"])
    edge_hits = 0
    for s in signals:
        idx = int(s["candle_index"])
        start = max(0, idx - 5)
        end = min(len(df), idx + 6)
        if s["direction"] == "sell":
            if highs.iloc[start:end].any():
                edge_hits += 1
        else:
            if lows.iloc[start:end].any():
                edge_hits += 1
    edge_accuracy = int(round(100 * edge_hits / count)) if count else 0

    # Persistence risk: price continues ≥2% in same direction within 24 candles
    persistence_fails = 0
    valid_persist = 0
    closes_arr = df["close"].to_numpy()
    for s in signals:
        idx = int(s["candle_index"])
        if idx + 24 >= len(closes_arr):
            continue
        valid_persist += 1
        price = closes_arr[idx]
        future = closes_arr[idx + 1 : idx + 25]
        if s["direction"] == "sell":
            if future.max() >= price * 1.02:
                persistence_fails += 1
        else:
            if future.min() <= price * 0.98:
                persistence_fails += 1
    persistence_fail_pct = (
        int(round(100 * persistence_fails / valid_persist)) if valid_persist else 0
    )

    # Cluster strength correlation: size vs slope quality (12-24 candles)
    sizes: List[float] = []
    qualities: List[float] = []
    for s in signals:
        idx = int(s["candle_index"])
        if idx + 24 >= len(closes_arr) or idx + 12 >= len(closes_arr):
            continue
        y = closes_arr[idx + 12 : idx + 25]
        if len(y) < 2:
            continue
        x_vals = np.arange(len(y))
        slope = float(np.polyfit(x_vals, y, 1)[0])
        direction_val = -1 if s["direction"] == "sell" else 1
        sizes.append(float(s["size"]))
        qualities.append(slope * direction_val)

    cluster_corr = 0.0
    if sizes and qualities:
        df_bins = pd.DataFrame({"size": sizes, "quality": qualities})
        try:
            df_bins["bin"] = pd.qcut(df_bins["size"], q=3, duplicates="drop")
            grouped = df_bins.groupby("bin").mean()
            if len(grouped) >= 2:
                cluster_corr = float(
                    np.corrcoef(grouped["size"], grouped["quality"])[0, 1]
                )
        except ValueError:
            cluster_corr = 0.0

    # Exhaustion-to-flip lag: distance to nearest Brain-2 flip
    from . import reversal
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        rev_signals = reversal.run(df, viz=False)
    rev_indices = [int(s["index"]) for s in rev_signals]
    lags: List[int] = []
    if rev_indices:
        for s in signals:
            idx = int(s["candle_index"])
            nearest = min(abs(idx - r) for r in rev_indices)
            lags.append(nearest)
    avg_lag = float(np.mean(lags)) if lags else 0.0

    # ------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------
    import sys

    timeframe = "unknown"
    if "--time" in sys.argv:
        idx = sys.argv.index("--time")
        if idx + 1 < len(sys.argv):
            timeframe = sys.argv[idx + 1]

    print(f"[BRAIN][exhaustion][{timeframe}]")
    print(f"  avg_uptrend_duration={avg_uptrend_duration:.2f}")
    print(f"  avg_downtrend_duration={avg_downtrend_duration:.2f}")
    print(f"  avg_up_reversal_slope24={avg_up_reversal_slope24:.5f}")
    print(f"  avg_down_reversal_slope24={avg_down_reversal_slope24:.5f}")
    print(f"  avg_uptrend_pressure={avg_uptrend_pressure:.2f}")
    print(f"  avg_downtrend_pressure={avg_downtrend_pressure:.2f}")

    print("[EXH][stats]")
    print(f"  Edge accuracy: {edge_accuracy}%")
    print(f"  Persistence fails: {persistence_fail_pct}%")
    print(f"  Cluster-size correlation: {cluster_corr:.2f}")
    print(f"  Exhaustion-to-flip lag: {avg_lag:.2f} candles")

    return {
        "brain": "exhaustion",
        "stats": {
            "count": count,
            "avg_gap": avg_gap,
            "slope_bias": f"{direction} {pct}%",
            "avg_uptrend_duration": avg_uptrend_duration,
            "avg_downtrend_duration": avg_downtrend_duration,
            "avg_up_reversal_slope24": avg_up_reversal_slope24,
            "avg_down_reversal_slope24": avg_down_reversal_slope24,
            "avg_uptrend_pressure": avg_uptrend_pressure,
            "avg_downtrend_pressure": avg_downtrend_pressure,
            "edge_accuracy": edge_accuracy,
            "persistence_fail_pct": persistence_fail_pct,
            "cluster_size_correlation": cluster_corr,
            "exhaustion_to_flip_lag": avg_lag,
        },
    }
