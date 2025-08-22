from __future__ import annotations

import argparse
import os
import sys
from statistics import mean

# Ensure repository root on path for direct execution
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from systems.tests.utils_truth import load_candles, slope

TAG = "SOLUSD"


def _trend_segments(prices):
    """Yield (direction, length, end_index) for each monotonic trend."""
    segments = []
    if len(prices) < 2:
        return segments
    direction = 0
    length = 1
    for i in range(1, len(prices)):
        diff = prices.iloc[i] - prices.iloc[i - 1]
        curr_dir = 1 if diff > 0 else -1 if diff < 0 else direction
        if curr_dir == direction:
            length += 1
        else:
            if direction != 0:
                segments.append((direction, length, i))
            direction = curr_dir
            length = 1
    if direction != 0:
        segments.append((direction, length, len(prices)))
    return segments


def _avg(values):
    return mean(values) if values else 0.0


def main(timeframe: str) -> None:
    file_path = f"data/sim/{TAG}_1h.csv"
    df = load_candles(file_path, timeframe)
    prices = df["close"]

    up_durations = []
    down_durations = []
    up_slopes = []
    down_slopes = []
    up_bubbles = []
    down_bubbles = []

    for direction, length, end in _trend_segments(prices):
        if length > 50:
            if direction == 1:
                up_durations.append(length)
            else:
                down_durations.append(length)
        if end >= 64 and end + 64 <= len(prices):
            prev = slope(prices.iloc[end - 64:end])
            nxt = slope(prices.iloc[end:end + 64])
            delta = nxt - prev
            if direction == 1:
                up_slopes.append(delta)
            else:
                down_slopes.append(delta)
        if direction == 1:
            up_bubbles.append(length)
        else:
            down_bubbles.append(length)

    print(
        f"[Exhaustion][Uptrend duration >50] → {_avg(up_durations):.1f} candles (N={len(up_durations)})"
    )
    print(
        f"[Exhaustion][Downtrend duration >50] → {_avg(down_durations):.1f} candles (N={len(down_durations)})"
    )
    print(
        f"[Exhaustion][Slope delta after uptrend reversal] → {_avg(up_slopes):.3f} (N={len(up_slopes)})"
    )
    print(
        f"[Exhaustion][Slope delta after downtrend reversal] → {_avg(down_slopes):.3f} (N={len(down_slopes)})"
    )
    print(
        f"[Exhaustion][Max bubble size before uptrend reversal] → {_avg(up_bubbles):.1f} (N={len(up_bubbles)})"
    )
    print(
        f"[Exhaustion][Max bubble size before downtrend reversal] → {_avg(down_bubbles):.1f} (N={len(down_bubbles)})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", default="1m")
    args = parser.parse_args()
    main(args.time)
