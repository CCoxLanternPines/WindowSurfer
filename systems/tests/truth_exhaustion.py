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
    percent_results,
    run_truth,
)

TAG = "SOLUSD"


QUESTIONS = [
    (
        "Does exhaustion after long pressure → reversal?",
        lambda t, df, ctx: ctx["is_exhaustion"][t]
        and slope(df["close"].iloc[t : t + 6]) < 0,
    ),
]


def build_context(df):
    """Precompute exhaustion markers and streak lengths."""
    streak = [0] * len(df)
    for i in range(1, len(df)):
        if df["close"].iloc[i] > df["close"].iloc[i - 1]:
            streak[i] = streak[i - 1] + 1
        else:
            streak[i] = 0

    is_exhaustion = [False] * len(df)
    for i in range(1, len(df)):
        is_exhaustion[i] = streak[i - 1] >= 5 and df["close"].iloc[i] < df["close"].iloc[i - 1]

    return {"streak": streak, "is_exhaustion": is_exhaustion}


def main(timeframe: str, vis: bool) -> None:
    if vis:
        run_simulation(timeframe=timeframe, viz=True)
        return

    df = load_candles(TAG, timeframe)
    results = run_truth(df, QUESTIONS, build_context)
    print(percent_results(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", default="1m")
    parser.add_argument("--vis", action="store_true", default=False)
    args = parser.parse_args()
    main(args.time, args.vis)

