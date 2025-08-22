from __future__ import annotations

import argparse
import os
import sys
import numpy as np

# Ensure repository root on path for direct execution
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from systems.sim_engine import run_simulation
from systems.tests.utils_truth import (
    load_candles,
    slope,
)


TAG = "SOLUSD"


QUESTIONS = [
    (
        "Average uptrend duration (>50 bars)",
        lambda t, df, ctx: (
            ctx["trend_len"][t]
            if ctx["is_exhaustion"][t]
            and ctx["trend_dir"][t] == "up"
            and ctx["trend_len"][t] > 50
            else None
        ),
    ),
    (
        "Average downtrend duration (>50 bars)",
        lambda t, df, ctx: (
            ctx["trend_len"][t]
            if ctx["is_exhaustion"][t]
            and ctx["trend_dir"][t] == "down"
            and ctx["trend_len"][t] > 50
            else None
        ),
    ),
    (
        "Slope delta after uptrend reversal (64 bars)",
        lambda t, df, ctx: (
            slope(df["close"].iloc[t : t + 64]) - slope(df["close"].iloc[t - 64 : t])
            if ctx["is_exhaustion"][t]
            and ctx["trend_dir"][t] == "up"
            and t >= 64
            and t + 64 < len(df)
            else None
        ),
    ),
    (
        "Slope delta after downtrend reversal (64 bars)",
        lambda t, df, ctx: (
            slope(df["close"].iloc[t : t + 64]) - slope(df["close"].iloc[t - 64 : t])
            if ctx["is_exhaustion"][t]
            and ctx["trend_dir"][t] == "down"
            and t >= 64
            and t + 64 < len(df)
            else None
        ),
    ),
    (
        "Max bubble size before uptrend reversal",
        lambda t, df, ctx: (
            ctx["pressure_len"][t]
            if ctx["is_exhaustion"][t] and ctx["trend_dir"][t] == "up"
            else None
        ),
    ),
    (
        "Max bubble size before downtrend reversal",
        lambda t, df, ctx: (
            ctx["pressure_len"][t]
            if ctx["is_exhaustion"][t] and ctx["trend_dir"][t] == "down"
            else None
        ),
    ),
]


def build_context(df):
    """Precompute exhaustion markers, pressure, volatility, and trend info."""
    prices = df["close"]

    # Track trend direction and length
    trend_dir = ["flat"] * len(df)
    trend_len = [0] * len(df)

    cur_dir = "flat"
    cur_len = 0
    for i in range(1, len(df)):
        if prices.iloc[i] > prices.iloc[i - 1]:
            if cur_dir == "up":
                cur_len += 1
            else:
                cur_dir = "up"
                cur_len = 1
        elif prices.iloc[i] < prices.iloc[i - 1]:
            if cur_dir == "down":
                cur_len += 1
            else:
                cur_dir = "down"
                cur_len = 1
        else:
            cur_dir = "flat"
            cur_len = 0

        trend_dir[i] = cur_dir
        trend_len[i] = cur_len

    # Compute exhaustion markers = end of long pressure
    pressure_len = [0] * len(df)
    is_exhaustion = [False] * len(df)
    pressure = 0
    for i in range(1, len(df)):
        if prices.iloc[i] > prices.iloc[i - 1]:
            pressure += 1
            is_exhaustion[i] = False
            pressure_len[i] = pressure
        elif prices.iloc[i] < prices.iloc[i - 1]:
            pressure_len[i] = pressure
            is_exhaustion[i] = pressure >= 5
            pressure = 0
        else:
            pressure_len[i] = pressure
            is_exhaustion[i] = False

    return {
        "is_exhaustion": is_exhaustion,
        "pressure_len": pressure_len,
        "trend_dir": trend_dir,
        "trend_len": trend_len,
    }


def run_truth(df, questions, build_context_fn):
    ctx = build_context_fn(df)
    results = {}
    for q, fn in questions:
        values = []
        for t in range(len(df)):
            try:
                val = fn(t, df, ctx)
                if val is not None:
                    values.append(val)
            except Exception:
                continue
        results[q] = values
    return results


def main(timeframe: str, vis: bool) -> None:
    if vis:
        run_simulation(timeframe=timeframe, viz=True)
        return

    file_path = f"data/sim/{TAG}_1h.csv"
    df = load_candles(file_path, timeframe)
    results = run_truth(df, QUESTIONS, build_context)

    for q, values in results.items():
        if not values:
            print(f"{q} → no samples")
            continue
        avg_val = np.mean(values)
        median_val = np.median(values)
        print(f"{q} → avg={avg_val:.3f}, median={median_val:.3f}, N={len(values)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", default="1m")
    parser.add_argument("--vis", action="store_true", default=False)
    args = parser.parse_args()
    main(args.time, args.vis)
