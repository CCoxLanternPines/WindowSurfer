from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


# Tuning constants
POSITION_BUY_MAX = 0.15
POSITION_SELL_MIN = 0.85
PROXIMITY = 0.015
PROMINENCE = 0.03
MIN_SPACING_HOURS = 240  # 10d
BUY_COOLDOWN_H = 120
SELL_COOLDOWN_H = 72


# ts, open, high, low, close
Candle = Tuple[int, float, float, float, float]


@dataclass
class Mark:
    ts: int
    price: float


def load_candles(tag: str) -> List[Candle]:
    """Load candles from ``data/raw/<TAG>.csv``.

    Each row in the CSV is expected to be: timestamp,open,high,low,close,volume.
    """

    path = Path("data/raw") / f"{tag}.csv"
    candles: List[Candle] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            candles.append(
                (
                    int(row["timestamp"]),
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                )
            )
    return candles


def run(tag: str, _: int = 0) -> None:
    candles = load_candles(tag)
    if len(candles) < 721:
        print("Not enough data to simulate.")
        return

    valleys_tradable: List[Mark] = []
    peaks_tradable: List[Mark] = []
    valleys_found = peaks_found = 0
    valleys_tradable_count = peaks_tradable_count = 0
    open_notes: List[Tuple[int, float]] = []

    buys = sells = wins = 0
    gain_total = 0.0
    hold_hours_total = 0.0
    buys_by_percentile = [0] * 10
    sells_by_percentile = [0] * 10
    blocked_by_downtrend_lock = 0

    next_buy_time = next_sell_time = 0
    last_valley_ts = last_peak_ts = -10**18

    for i in range(720, len(candles)):
        ts, o, h, l, c = candles[i]

        window = candles[i - 720 + 1 : i + 1]
        low_30d = min(x[3] for x in window)
        high_30d = max(x[2] for x in window)
        position = (c - low_30d) / max(1e-9, high_30d - low_30d)

        prev_closes_10d = [x[4] for x in candles[max(0, i - 240) : i]]
        if c == low_30d:
            valleys_found += 1
            if (
                prev_closes_10d
                and c < min(prev_closes_10d)
                and (max(prev_closes_10d) - c) / c >= PROMINENCE
                and ts - last_valley_ts >= MIN_SPACING_HOURS * 3600
            ):
                valleys_tradable.append(Mark(ts, c))
                valleys_tradable_count += 1
                last_valley_ts = ts
        if c == high_30d:
            peaks_found += 1
            if (
                prev_closes_10d
                and c > max(prev_closes_10d)
                and (c - min(prev_closes_10d)) / c >= PROMINENCE
                and ts - last_peak_ts >= MIN_SPACING_HOURS * 3600
            ):
                peaks_tradable.append(Mark(ts, c))
                peaks_tradable_count += 1
                last_peak_ts = ts

        # Sells
        if open_notes and position >= POSITION_SELL_MIN:
            if ts >= next_sell_time:
                near_peak = any(
                    abs(c - p.price) / p.price <= PROXIMITY for p in peaks_tradable
                )
                if near_peak:
                    idx = min(9, int(position * 10))
                    sells_by_percentile[idx] += len(open_notes)
                    next_sell_time = ts + SELL_COOLDOWN_H * 3600
                    for entry_time, entry_price in open_notes:
                        gain_pct = (c - entry_price) / entry_price * 100
                        hold_hours = (ts - entry_time) / 3600
                        sells += 1
                        gain_total += gain_pct
                        hold_hours_total += hold_hours
                        if gain_pct > 0:
                            wins += 1
                        print(f"[SELL] {gain_pct:+.2f}% in {hold_hours/24:.1f} days")
                    open_notes = []

        # Buys
        if position <= POSITION_BUY_MAX:
            if ts >= next_buy_time:
                near_valley = any(
                    abs(c - v.price) / v.price <= PROXIMITY for v in valleys_tradable
                )
                if near_valley:
                    close_30d_start = candles[i - 720][4]
                    close_90d_start = candles[i - 2160][4] if i >= 2160 else candles[0][4]
                    slope_30d = c - close_30d_start
                    slope_90d = c - close_90d_start
                    if not (slope_30d < 0 and slope_90d < 0):
                        prev_c = candles[i - 1][4]
                        hl2 = (h + l) / 2
                        prev_hl2 = (candles[i - 1][2] + candles[i - 1][3]) / 2
                        if c > prev_c and hl2 > prev_hl2:
                            open_notes.append((ts, c))
                            buys += 1
                            idx = min(9, int(position * 10))
                            buys_by_percentile[idx] += 1
                            next_buy_time = ts + BUY_COOLDOWN_H * 3600
                            print(f"[BUY] ${c:.2f} at {ts}")
                    else:
                        blocked_by_downtrend_lock += 1

    avg_gain = gain_total / sells if sells else 0.0
    avg_hold_days = (hold_hours_total / sells / 24) if sells else 0.0
    win_rate = (wins / sells * 100) if sells else 0.0

    print(
        f"[SUMMARY] Buys: {buys} | Sells: {sells} | WinRate: {win_rate:.2f}% | AvgGain: {avg_gain:.2f}% | AvgHold: {avg_hold_days:.1f} days"
    )
    print(
        f"[SUMMARY] Peaks: {peaks_found} (tradable {peaks_tradable_count}) | Valleys: {valleys_found} (tradable {valleys_tradable_count})"
    )
    print(
        f"[SUMMARY] Downtrend blocks: {blocked_by_downtrend_lock} | BuyDist: {buys_by_percentile} | SellDist: {sells_by_percentile}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Basic monthly wave simulation.")
    parser.add_argument("tag", help="Asset tag for CSV in data/raw/<TAG>.csv")
    parser.add_argument("_extra", nargs="?", default=0)
    args = parser.parse_args()
    run(args.tag, args._extra)


if __name__ == "__main__":
    main()
