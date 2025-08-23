"""CLI for brain training and validation.

This script provides brain-first workflows for auditing, teaching, and
future corrections of trading brains. It dynamically loads brain modules
from ``systems.brains`` and works with simple CSV candle data and
append-only JSONL labels.
"""

from __future__ import annotations

import argparse
import json
import importlib
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


DATA_DIR = Path("data")
CANDLES_DIR = DATA_DIR / "candles" / "sim"
LABELS_DIR = DATA_DIR / "labels"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_candles(symbol: str, timeframe: str) -> pd.DataFrame:
    """Load candle data from CSV and ensure ``candle_index`` column."""
    path = CANDLES_DIR / f"{symbol}_{timeframe}.csv"
    df = pd.read_csv(path)
    if "candle_index" not in df.columns:
        df = df.reset_index().rename(columns={"index": "candle_index"})
    return df


def load_labels(brain: str) -> list[dict]:
    path = LABELS_DIR / f"{brain}.jsonl"
    labels: list[dict] = []
    if path.exists():
        with path.open() as fh:
            for line in fh:
                try:
                    labels.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return labels


def append_labels(brain: str, records: list[dict]) -> None:
    if not records:
        return
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    path = LABELS_DIR / f"{brain}.jsonl"
    with path.open("a") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def audit(brain_mod, df: pd.DataFrame) -> None:
    signals = brain_mod.run(df, viz=False)
    stats = brain_mod.summarize(signals, df)
    print(json.dumps(stats, indent=2))


def teach(brain_mod, df: pd.DataFrame, brain_name: str) -> None:
    signals = brain_mod.run(df, viz=False)
    existing = load_labels(brain_name)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["candle_index"], df["close"], label="Close")

    buy_x = [s["candle_index"] for s in signals if s["direction"] == "buy"]
    buy_y = [s["price"] for s in signals if s["direction"] == "buy"]
    sell_x = [s["candle_index"] for s in signals if s["direction"] == "sell"]
    sell_y = [s["price"] for s in signals if s["direction"] == "sell"]
    ax.scatter(buy_x, buy_y, c="green", marker="^", label="brain buy")
    ax.scatter(sell_x, sell_y, c="red", marker="v", label="brain sell")

    label_color = {"BUY": "green", "SELL": "red", "HOLD": "yellow"}
    for lbl in existing:
        ax.scatter([lbl["idx"]], [lbl["price"]], c=label_color.get(lbl["label"], "blue"), marker="x")

    ax.set_title(f"Teaching brain: {brain_name}")
    ax.set_xlabel("candle index")
    ax.set_ylabel("price")
    ax.grid(True)
    ax.legend()

    new_labels: list[dict] = []

    def on_key(event):
        if event.key == "escape":
            plt.close(event.canvas.figure)
            return
        key_map = {"b": "BUY", "s": "SELL", "h": "HOLD"}
        if event.key not in key_map:
            return
        if event.xdata is None or event.ydata is None:
            return
        idx = int(event.xdata)
        price = float(event.ydata)
        label = key_map[event.key]
        new_labels.append({"idx": idx, "price": price, "label": label})
        ax.scatter([idx], [price], c=label_color[label], marker="o")
        fig.canvas.draw()

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

    append_labels(brain_name, new_labels)


def correct() -> None:
    print("[TODO] Correction mode not yet implemented.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Brain training and validation CLI")
    parser.add_argument("--brain", required=True, help="Brain module name (e.g., exhaustion)")
    parser.add_argument("--symbol", required=True, help="Market symbol (e.g., DOGEUSD)")
    parser.add_argument("--time", required=True, help="Timeframe (e.g., 1m, 1w)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit", action="store_true", help="Run brain and print stats")
    mode.add_argument("--teach", action="store_true", help="Interactive labeling session")
    mode.add_argument("--correct", action="store_true", help="Placeholder for corrections")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    df = load_candles(args.symbol, args.time)
    brain_mod = importlib.import_module(f"systems.brains.{args.brain}")

    if args.audit:
        audit(brain_mod, df)
    elif args.teach:
        teach(brain_mod, df, args.brain)
    else:
        correct()


if __name__ == "__main__":
    main()
