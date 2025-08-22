from __future__ import annotations

import argparse
import os
import sys

# Ensure repository root on path for direct execution
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from systems.sim_engine import run_simulation
from systems.tests.utils_truth import (
    load_candles,
    slope,
    run_truth,
)

TAG = "SOLUSD"
VOLATILITY_THRESH = 0.02


QUESTIONS = [
    (
        "Trend vs noise: exhaustion fires more in chop?",
        lambda t, df, ctx: (
            ctx["volatility"][t] > ctx["volatility_thresh"]
            if ctx["is_exhaustion"][t]
            else None
        ),
    ),
    (
        "Reversal trigger: first exhaustion after long pressure → reversal?",
        lambda t, df, ctx: (
            slope(df["close"].iloc[t : t + 6]) < 0
            if ctx["is_exhaustion"][t] and ctx["pressure_len"][t] > 8
            else None
        ),
    ),
    (
        "Nervousness factor: longer since last exhaustion → higher reversal chance?",
        lambda t, df, ctx: (
            slope(df["close"].iloc[t : t + 6]) < 0
            if ctx["is_exhaustion"][t] and ctx["bars_since_exh"][t] > 10
            else None
        ),
    ),
    (
        "Slope decay: after long uptrend, 64-candle slope weakens after exhaustion?",
        lambda t, df, ctx: (
            slope(df["close"].iloc[t - 64 : t]) > 0
            and slope(df["close"].iloc[t : t + 64])
            < slope(df["close"].iloc[t - 64 : t])
            if (
                ctx["is_exhaustion"][t]
                and ctx["pressure_len"][t] > 8
                and t >= 64
                and t + 64 < len(df)
            )
            else None
        ),
    ),
    (
        "Bubble size: larger exhaustion bubble → higher reversal chance?",
        lambda t, df, ctx: (
            slope(df["close"].iloc[t : t + 64]) < 0
            if (
                ctx["is_exhaustion"][t]
                and ctx["pressure_len"][t] > 12
                and t + 64 < len(df)
            )
            else None
        ),
    ),
]


def build_context(df):
    """Precompute exhaustion markers, pressure, and volatility."""
    prices = df["close"]

    # Rolling volatility as coefficient of variation.
    rolling_std = prices.rolling(window=20, min_periods=1).std()
    rolling_mean = prices.rolling(window=20, min_periods=1).mean()
    volatility = (rolling_std / rolling_mean).fillna(0).tolist()

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

    bars_since_exh = [0] * len(df)
    last = -1
    for i in range(len(df)):
        if is_exhaustion[i]:
            bars_since_exh[i] = 0
            last = i
        else:
            bars_since_exh[i] = i - last if last != -1 else i + 1

    return {
        "is_exhaustion": is_exhaustion,
        "pressure_len": pressure_len,
        "volatility": volatility,
        "volatility_thresh": VOLATILITY_THRESH,
        "bars_since_exh": bars_since_exh,
    }


def main(timeframe: str, vis: bool) -> None:
    if vis:
        run_simulation(timeframe=timeframe, viz=True)
        return

    file_path = f"data/sim/{TAG}_1h.csv"
    df = load_candles(file_path, timeframe)
    results = run_truth(df, QUESTIONS, build_context)
    for q, (hits, total) in results.items():
        pct = (hits / total * 100) if total else 0.0
        print(f"{q} → ({hits}/{total}) {pct:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", default="1m")
    parser.add_argument("--vis", action="store_true", default=False)
    args = parser.parse_args()
    main(args.time, args.vis)

