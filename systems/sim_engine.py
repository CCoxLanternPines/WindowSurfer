from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple


Candle = Tuple[int, float]


def load_candles(tag: str) -> List[Candle]:
    """Load candles from ``data/raw/<TAG>.csv``.

    Each row in the CSV is expected to be: timestamp,open,high,low,close,volume.
    Only the timestamp and close columns are used.
    """

    path = Path("data/raw") / f"{tag}.csv"
    candles: List[Candle] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            candles.append((int(row["timestamp"]), float(row["close"])) )
    return candles


def run(tag: str) -> None:
    candles = load_candles(tag)
    if len(candles) < 721:
        print("Not enough data to simulate.")
        return

    valleys: List[Candle] = []
    peaks: List[Candle] = []
    open_notes: List[Candle] = []

    buys = sells = wins = 0
    gain_total = 0.0
    hold_hours_total = 0.0

    for i in range(720, len(candles)):
        window = candles[i - 720 : i + 1]
        close_prices = [c[1] for c in window]
        slope = close_prices[-1] - close_prices[0]

        recent = close_prices[-360:-1]
        if recent:
            if close_prices[-1] == min(close_prices) and close_prices[-1] < min(recent):
                valleys.append((candles[i][0], close_prices[-1]))
            if close_prices[-1] == max(close_prices) and close_prices[-1] > max(recent):
                peaks.append((candles[i][0], close_prices[-1]))

        price = close_prices[-1]
        ts = candles[i][0]

        if slope >= 0 and any(abs(price - v_price) / v_price <= 0.05 for _, v_price in valleys):
            open_notes.append((ts, price))
            buys += 1
            print(f"[BUY] ${price:.2f} at {ts}")

        remaining: List[Candle] = []
        for entry_time, entry_price in open_notes:
            if any(abs(price - p_price) / p_price <= 0.05 for _, p_price in peaks):
                gain_pct = (price - entry_price) / entry_price * 100
                hold_hours = (ts - entry_time) / 3600
                sells += 1
                gain_total += gain_pct
                hold_hours_total += hold_hours
                if gain_pct > 0:
                    wins += 1
                print(f"[SELL] {gain_pct:+.2f}% in {hold_hours/24:.1f} days")
            else:
                remaining.append((entry_time, entry_price))
        open_notes = remaining

    avg_gain = gain_total / sells if sells else 0.0
    avg_hold_days = (hold_hours_total / sells / 24) if sells else 0.0
    win_rate = (wins / sells * 100) if sells else 0.0

    print("\n--- Summary ---")
    print(f"Buys: {buys}")
    print(f"Sells: {sells}")
    print(f"Win rate: {win_rate:.2f}%")
    print(f"Avg gain: {avg_gain:.2f}%")
    print(f"Avg hold: {avg_hold_days:.1f} days")


def main() -> None:
    parser = argparse.ArgumentParser(description="Basic monthly wave simulation.")
    parser.add_argument("tag", help="Asset tag for CSV in data/raw/<TAG>.csv")
    args = parser.parse_args()
    run(args.tag)


if __name__ == "__main__":
    main()
