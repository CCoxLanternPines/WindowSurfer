from __future__ import annotations

import argparse
import os
import sys

# Ensure repository root on path for direct execution
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from systems.sim_engine import run_simulation
from systems.tests.utils_truth import load_candles, slope, percent_results

TAG = "SOLUSD"

QUESTIONS = [
    ("Is 5-candle slope positive?",
     lambda df, i: slope(df['close'].iloc[max(0, i-4):i+1]) > 0),
    ("Is close above previous close?",
     lambda df, i: True if i == 0 else df['close'].iloc[i] > df['close'].iloc[i-1]),
]


def run_truth(timeframe: str, vis: bool) -> None:
    if vis:
        run_simulation(timeframe=timeframe, viz=True)
        return

    df = load_candles(TAG, timeframe)
    results = {q: [] for q, _ in QUESTIONS}
    for i in range(len(df)):
        for q, fn in QUESTIONS:
            try:
                results[q].append(bool(fn(df, i)))
            except Exception:
                results[q].append(False)

    print(percent_results(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", default="1m")
    parser.add_argument("--vis", action="store_true", default=False)
    args = parser.parse_args()
    run_truth(args.time, args.vis)
