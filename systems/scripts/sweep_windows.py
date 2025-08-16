from __future__ import annotations

"""Sweep simulation accuracy across window durations."""

import contextlib
import csv
import os
import re
from io import StringIO
from typing import Dict, List

import matplotlib.pyplot as plt

from systems import sim_engine


DURATIONS = ["1d", "3d", "1w", "2w", "1m", "3m"]


def parse_duration(duration: str) -> int:
    """Return the candle count for ``duration`` assuming 1h candles."""
    match = re.match(r"(\d+)([dwm])", duration)
    if not match:
        raise ValueError(f"Unsupported duration: {duration}")
    value = int(match.group(1))
    unit = match.group(2)
    hours_per = {"d": 24, "w": 7 * 24, "m": 30 * 24}
    return value * hours_per[unit]


def run_window_sweep() -> None:
    """Run the sweep across predefined durations and record results."""
    results: List[Dict[str, float]] = []
    csv_path = os.path.join("results", "window_sweep.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    for dur in DURATIONS:
        candles = parse_duration(dur)
        sim_engine.DEFAULT_BOTTOM_WINDOW = candles

        buf = StringIO()
        original_show = sim_engine.plt.show
        sim_engine.plt.show = lambda: None
        try:
            with contextlib.redirect_stdout(buf):
                sim_engine.run_simulation(timeframe="")
        finally:
            sim_engine.plt.show = original_show
        output = buf.getvalue()

        match = re.search(
            r"Raw Accuracy: ([0-9.]+)% \| Weighted: ([0-9.]+)%",
            output,
        )
        if not match:
            raise RuntimeError("Failed to parse accuracy from simulation output")
        accuracy = float(match.group(1))
        weighted = float(match.group(2))

        results.append(
            {
                "duration": dur,
                "candles": candles,
                "accuracy": accuracy,
                "weighted_accuracy": weighted,
            }
        )

    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, ["duration", "candles", "accuracy", "weighted_accuracy"]
        )
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)

    print(f"{'Dur':<5} {'Candles':>7} {'Accuracy':>10} {'Weighted':>10}")
    for row in results:
        print(
            f"{row['duration']:<5} {row['candles']:>7} {row['accuracy']:>9.2f}% {row['weighted_accuracy']:>9.2f}%"
        )

    durations = [r["duration"] for r in results]
    acc_vals = [r["accuracy"] for r in results]
    weighted_vals = [r["weighted_accuracy"] for r in results]
    x = range(len(durations))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([xi - width / 2 for xi in x], acc_vals, width, label="Accuracy")
    ax.bar([xi + width / 2 for xi in x], weighted_vals, width, label="Weighted")
    ax.set_xticks(list(x))
    ax.set_xticklabels(durations)
    ax.set_ylabel("Percent")
    ax.set_title("Forecast Accuracy by Window")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_window_sweep()
