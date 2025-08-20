from __future__ import annotations

"""Plot structured trading logs with hover annotations."""

import argparse
import json
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import mplcursors
import pandas as pd

from systems.utils.time import parse_duration


def view_log(account_name: str, timeframe: str | None = None) -> None:
    """Render a scatter plot of decisions from ``account_name`` log."""

    log_path = Path(f"data/logs/{account_name}.json")
    if not log_path.exists():
        print(f"[ERROR] No log found at {log_path}")
        return

    with log_path.open() as f:
        events = json.load(f)

    if timeframe:
        delta = parse_duration(timeframe)
        cutoff = datetime.utcnow() - delta
        events = [e for e in events if pd.to_datetime(e["timestamp"]) >= cutoff]

    if not events:
        print(f"[EMPTY] {account_name} log has no entries")
        return

    times = pd.to_datetime([e["timestamp"] for e in events])
    prices: list[float] = []
    colors: list[str] = []
    annotations: list[str] = []

    for e in events:
        decision = e["decision"]
        trades = e.get("trades") or []
        if trades:
            price = trades[0].get("price")
        else:
            # fallback to candle close if no trade
            price = e.get("features", {}).get("close")
        if price is None:
            price = 0.0

        prices.append(price)

        if decision == "BUY":
            colors.append("green")
        elif decision == "SELL":
            colors.append("red")
        elif decision == "FLAT":
            colors.append("orange")
        elif decision == "HOLD":
            colors.append("gray")
        else:
            colors.append("yellow")  # catch unexpected cases

        features = e.get("features", {})
        annotations.append(
            f"{e['timestamp']}\n"
            f"Decision: {decision}\n"
            f"Slope: {features.get('slope')}\n"
            f"Volatility: {features.get('volatility')}\n"
            f"BuyP: {features.get('buy_pressure')} | "
            f"SellP: {features.get('sell_pressure')}"
        )

    fig, ax = plt.subplots()
    sc = ax.scatter(times, prices, c=colors, marker="o")

    cursor = mplcursors.cursor(sc, hover=True)

    @cursor.connect("add")
    def on_hover(sel):
        sel.annotation.set(text=annotations[sel.index])

    ax.set_title(f"Trading Decisions for {account_name}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price (USDT)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--account", required=True)
    parser.add_argument("--time", type=str, default=None)
    args = parser.parse_args()
    view_log(args.account, timeframe=args.time)

