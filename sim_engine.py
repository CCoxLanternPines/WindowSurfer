from __future__ import annotations

from collections import deque
import csv
from pathlib import Path
from typing import Dict

def run_sim(config: Dict[str, int | float]) -> Dict[str, float]:
    """Run a minimal trading simulation and return metrics.

    The simulation loads historical candle data from ``data/raw/DOGEUSD.csv``
    and executes a simple strategy based on the provided configuration.
    """
    capital = float(config["simulation_capital"])
    buy_frequency = int(config["buy_frequency"])
    min_hold_time = int(config["min_hold_time"])
    avg_window = int(config["avg_window_all_time"])

    data_path = Path("data/raw/DOGEUSD.csv")
    prices: list[float] = []
    with data_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prices.append(float(row["close"]))

    window: deque[float] = deque(maxlen=avg_window)
    holding = False
    entry_price = 0.0
    entry_step = 0
    coin_amount = 0.0

    for step, price in enumerate(prices):
        window.append(price)
        if not holding:
            if step % buy_frequency == 0 and len(window) == avg_window:
                avg_price = sum(window) / avg_window
                if price < avg_price and capital > 0:
                    coin_amount = capital / price
                    capital = 0.0
                    holding = True
                    entry_price = price
                    entry_step = step
        else:
            if step - entry_step >= min_hold_time and price > entry_price:
                capital = coin_amount * price
                coin_amount = 0.0
                holding = False

    if holding and prices:
        capital += coin_amount * prices[-1]

    final_capital = capital
    pnl = final_capital - float(config["simulation_capital"])
    return {"final_capital": final_capital, "pnl": pnl}
