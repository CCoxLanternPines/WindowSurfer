from __future__ import annotations
import argparse, csv
from pathlib import Path
from typing import List, Tuple

# ts, open, high, low, close
Candle = Tuple[int, float, float, float, float]
LOG_PATH = Path("data/tmp/snapshots.log")

def load_candles(tag: str) -> List[Candle]:
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

def init_snapshot_log():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("")  # wipe on start

def log_snapshot(line: str, also_print: bool = True):
    with LOG_PATH.open("a") as f:
        f.write(line + "\n")
    if also_print:
        print(line)

def run(tag: str):
    candles = load_candles(tag)
    if len(candles) < 721:
        print("Not enough data to simulate.")
        return

    init_snapshot_log()

    WINDOW = 132   
    STEP = 24  

    zone_history = []
    avgpos_history = []

    for i in range(WINDOW, len(candles), STEP):
        window = candles[i - WINDOW : i]
        lows = [x[3] for x in window]
        highs = [x[2] for x in window]
        closes = [x[4] for x in window]

        low_30d = min(lows)
        high_30d = max(highs)

        positions = [
            (c - low_30d) / (high_30d - low_30d) if high_30d != low_30d else 0.5
            for c in closes
        ]
        avg_pos = sum(positions) / len(positions)

        current_close = closes[-1]
        pos_now = (
            (current_close - low_30d) / (high_30d - low_30d)
            if high_30d != low_30d
            else 0.5
        )

        # Zone classification
        if pos_now <= 0.2:
            zone = "Bottom zone"
        elif pos_now >= 0.8:
            zone = "Top zone"
        else:
            zone = "Mid zone"

        # Days in zone
        if zone_history and zone_history[-1] == zone:
            days_in_zone = (len(zone_history) + 1) * (STEP / 24)
        else:
            days_in_zone = STEP / 24

        zone_history.append(zone)
        avgpos_history.append(avg_pos)

        # Entry speed (% per day from opposite range)
        if zone == "Top zone":
            ref_price = low_30d
        elif zone == "Bottom zone":
            ref_price = high_30d
        else:
            ref_price = None

        if ref_price and ref_price != 0:
            pct_move = abs((current_close - ref_price) / ref_price) * 100
            entry_speed = pct_move / days_in_zone
        else:
            entry_speed = 0

        # AvgPos trend (last 3 snapshots)
        if len(avgpos_history) >= 3:
            trend_val = avgpos_history[-1] - avgpos_history[-3]
            if trend_val > 0.02:
                avgpos_trend = "rising"
            elif trend_val < -0.02:
                avgpos_trend = "falling"
            else:
                avgpos_trend = "flat"
        else:
            avgpos_trend = "flat"

        # Scenario tag
        if zone == "Bottom zone" and avgpos_trend == "rising" and entry_speed > 1:
            scenario = "BUY probability high"
        elif zone == "Top zone" and avgpos_trend == "falling" and entry_speed > 1:
            scenario = "SELL probability high"
        else:
            scenario = "No strong bias"

        ts = window[-1][0]
        line = (
            f"[SNAPSHOT] {ts} | AvgPos: {avg_pos:.2f} | PosNow: {pos_now:.2f} | "
            f"DaysInZone: {days_in_zone:.1f} | EntrySpeed: {entry_speed:.2f}%/day | "
            f"Bias: {avgpos_trend} | Zone: {zone} | {scenario}"
        )
        log_snapshot(line)

def main():
    parser = argparse.ArgumentParser(description="Monthly wave snapshot with predictive tags.")
    parser.add_argument("tag", help="Asset tag for CSV in data/raw/<TAG>.csv")
    args = parser.parse_args()
    run(args.tag)

if __name__ == "__main__":
    main()
