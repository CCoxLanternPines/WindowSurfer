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
        lambda t, df, ctx: (
            slope(df["close"].iloc[t : t + 6]) < 0
            if ctx["is_exhaustion"][t]
            else None
        ),
    ),
    (
        "Do long gaps without exhaustion align with trending?",
        lambda t, df, ctx: (
            abs(slope(df["close"].iloc[max(0, t - 12) : t])) > 0.01
            if ctx["since_exhaustion"][t] >= 24
            else None
        ),
    ),
]


def build_context(df):
    """Precompute exhaustion markers and streak / gap lengths."""
    streak = [0] * len(df)
    for i in range(1, len(df)):
        streak[i] = streak[i - 1] + 1 if df["close"].iloc[i] > df["close"].iloc[i - 1] else 0

    is_exhaustion = [False] * len(df)
    for i in range(1, len(df)):
        is_exhaustion[i] = streak[i - 1] >= 5 and df["close"].iloc[i] < df["close"].iloc[i - 1]

    since_exhaustion = [0] * len(df)
    last = -1
    for i in range(len(df)):
        if is_exhaustion[i]:
            since_exhaustion[i] = 0
            last = i
        else:
            since_exhaustion[i] = i - last if last != -1 else i + 1

    return {"streak": streak, "is_exhaustion": is_exhaustion, "since_exhaustion": since_exhaustion}


def main(timeframe: str, vis: bool) -> None:
    if vis:
        run_simulation(timeframe=timeframe, viz=True)
        return

    file_path = f"data/sim/{TAG}_1h.csv"
    df = load_candles(file_path, timeframe)
    results = run_truth(df, QUESTIONS, build_context)
    for q, pct in percent_results(results).items():
        print(f"{q} = {pct:.0f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", default="1m")
    parser.add_argument("--vis", action="store_true", default=False)
    args = parser.parse_args()
    main(args.time, args.vis)

