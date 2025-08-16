from __future__ import annotations

"""Very small historical simulation engine."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import pandas as pd

from .scripts import evaluate_buy, evaluate_sell


def run_simulation(*, timeframe: str = "1m") -> None:
    """Run a simple simulation over SOLUSD candles."""
    csv_path = Path("data/sim/SOLUSD.csv")
    df = pd.read_csv(csv_path)
    df.columns = [str(c).lower().strip() for c in df.columns]

    aliases = {
        "timestamp": ["timestamp", "time", "date"],
        "open": ["open", "o"],
        "high": ["high", "h"],
        "low": ["low", "l"],
        "close": ["close", "c", "close_price"],
        "volume": ["volume", "v"],
    }

    resolved = {}
    for target, opts in aliases.items():
        for col in opts:
            if col in df.columns:
                resolved[col] = target
                break

    missing = [t for t in aliases.keys() if t not in resolved.values()]
    if missing:
        raise ValueError(
            "Missing required candle columns: "
            f"{missing}. Found columns: {list(df.columns)}"
        )

    df = df.rename(columns=resolved)[list(aliases.keys())]

    if timeframe == "1m":
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=30)
        df = df[df["timestamp"] >= int(cutoff.timestamp())]

    df["short"] = df["close"].rolling(window=10, min_periods=1).mean()
    df["long"] = df["close"].rolling(window=50, min_periods=1).mean()
    df["delta"] = df["short"] - df["long"]

    state: Dict[str, Any] = {}
    for _, candle in df.iterrows():
        evaluate_buy.evaluate_buy(candle.to_dict(), state)
        evaluate_sell.evaluate_sell(candle.to_dict(), state)

    out_path = Path("data/tmp/discovery.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(df["timestamp"], df["close"], label="Close")
    plt.plot(df["timestamp"], df["delta"], label="Delta")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
