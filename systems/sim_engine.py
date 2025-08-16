from __future__ import annotations

"""Very small historical simulation engine."""

from datetime import datetime, timedelta
from typing import Dict, Any

import matplotlib.pyplot as plt
import pandas as pd

from .scripts import evaluate_buy, evaluate_sell


def run_simulation(*, timeframe: str = "1m") -> None:
    """Run a simple simulation over SOLUSD candles."""
    file_path = "data/sim/SOLUSD_1h.csv"
    df = pd.read_csv(file_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

    if timeframe == "1m":
        cutoff = datetime.utcnow() - timedelta(days=30)
        df = df[df["timestamp"] >= cutoff]

    df["short"] = df["close"].rolling(window=10, min_periods=1).mean()
    df["long"] = df["close"].rolling(window=50, min_periods=1).mean()
    df["delta"] = df["short"] - df["long"]

    delta = df["delta"]
    close = df["close"]
    # scale delta into the price range
    scale = (close.max() - close.min()) / (delta.max() - delta.min())
    norm_delta = (delta - delta.min()) * scale + close.min()
    df["norm_delta"] = norm_delta

    state: Dict[str, Any] = {}
    for _, candle in df.iterrows():
        evaluate_buy.evaluate_buy(candle.to_dict(), state)
        evaluate_sell.evaluate_sell(candle.to_dict(), state)

    plt.figure(figsize=(12, 6))
    plt.plot(df["timestamp"], df["close"], label="Close Price", color="blue")
    plt.plot(df["timestamp"], df["norm_delta"], label="Delta (scaled)", color="red")
    plt.xlabel("Time")
    plt.ylabel("Price / Scaled Delta")
    plt.title("SOLUSD Discovery Simulation")
    plt.legend()
    plt.grid(True)
    plt.show()
