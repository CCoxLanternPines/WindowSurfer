from __future__ import annotations

"""Very small historical simulation engine."""

import re
from datetime import timedelta
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd

from .scripts import evaluate_buy, evaluate_sell


# === Regime Detection Settings ===
SHORT_MA = 10   # short moving average lookback (candles)
LONG_MA  = 50   # long moving average lookback (candles)


def parse_timeframe(tf: str) -> timedelta | None:
    match = re.match(r"(\d+)([dhmw])", tf)
    if not match:
        return None
    val, unit = int(match.group(1)), match.group(2)
    if unit == "d":
        return timedelta(days=val)
    if unit == "w":
        return timedelta(weeks=val)
    if unit == "m":
        return timedelta(days=30 * val)  # rough month
    if unit == "h":
        return timedelta(hours=val)
    return None


def run_simulation(*, timeframe: str = "1m") -> None:
    """Run a simple simulation over SOLUSD candles."""
    file_path = "data/sim/SOLUSD_1h.csv"
    df = pd.read_csv(file_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

    if timeframe:
        delta = parse_timeframe(timeframe)
        if delta:
            cutoff = pd.Timestamp.utcnow().tz_localize(None) - delta
            df = df[df["timestamp"] >= cutoff]

    df["short"] = df["close"].rolling(window=SHORT_MA, min_periods=1).mean()
    df["long"] = df["close"].rolling(window=LONG_MA, min_periods=1).mean()
    df["delta"] = df["short"] - df["long"]

    # scale delta into the price range
    scale = (df["close"].max() - df["close"].min()) / (
        df["delta"].max() - df["delta"].min()
    )
    df["norm_delta"] = (
        (df["delta"] - df["delta"].min()) * scale + df["close"].min()
    )

    state: Dict[str, Any] = {}
    for _, candle in df.iterrows():
        evaluate_buy.evaluate_buy(candle.to_dict(), state)
        evaluate_sell.evaluate_sell(candle.to_dict(), state)

    plt.figure(figsize=(12, 6))
    plt.plot(df["timestamp"], df["close"], label="Close Price", color="blue")
    plt.plot(
        df["timestamp"],
        df["norm_delta"],
        label=f"Delta ({SHORT_MA}-{LONG_MA})",
        color="red",
    )
    plt.xlabel("Time")
    plt.ylabel("Price / Scaled Delta")
    plt.title("SOLUSD Discovery Simulation")
    plt.legend()
    plt.grid(True)
    plt.show()
