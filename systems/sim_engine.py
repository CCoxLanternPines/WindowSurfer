from __future__ import annotations
import argparse, csv
from pathlib import Path
from typing import List, Tuple

# ts, open, high, low, close
Candle = Tuple[int, float, float, float, float]
LOG_PATH = Path("data/tmp/snapshots.log")

# Window and step sizes
WINDOW = 132
STEP = 24

# --- Top/Bottom score knobs ---
ALPHA_WICK = 0.12     # how strongly wicks skew PosNow toward 0/1
SMOOTH_EMA = 0.25     # 0 disables smoothing; else EMA factor in (0,1]


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


def init_snapshot_log() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("")  # wipe on start


def log_snapshot(line: str, also_print: bool = True) -> None:
    with LOG_PATH.open("a") as f:
        f.write(line + "\n")
    if also_print:
        print(line)


def clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def run(tag: str) -> None:
    candles = load_candles(tag)
    if len(candles) < WINDOW:
        print("Not enough data to simulate.")
        return

    init_snapshot_log()
    prev_tb = 0.5

    for i in range(WINDOW, len(candles), STEP):
        window = candles[i - WINDOW : i]
        lows = [x[3] for x in window]
        highs = [x[2] for x in window]
        closes = [x[4] for x in window]

        low_w = min(lows)
        high_w = max(highs)
        denom = max(1e-9, high_w - low_w)

        pos_now = 0.5 if high_w == low_w else (closes[-1] - low_w) / denom
        pos_now = clip(pos_now, 0.0, 1.0)

        norms = [(c - low_w) / denom for c in closes] if high_w != low_w else [0.5] * len(closes)
        avg_pos = clip(sum(norms) / len(norms), 0.0, 1.0)

        # Wick features from the last candle
        last_ts, o, h, l, c = window[-1]
        rng = max(1e-9, h - l)
        body_low, body_high = (min(o, c), max(o, c))
        lower_wick = body_low - l      # rejection off lows
        upper_wick = h - body_high     # rejection off highs
        lw_ratio = clip(lower_wick / rng, 0.0, 1.0)
        uw_ratio = clip(upper_wick / rng, 0.0, 1.0)

        # Base top/bottom = normalized position
        topbottom_raw = pos_now

        # Symmetric wick skew: more upper wick ⇒ push toward top (↑),
        # more lower wick ⇒ push toward bottom (↓)
        wick_skew = ALPHA_WICK * (uw_ratio - lw_ratio)
        topbottom = clip(topbottom_raw + wick_skew, 0.0, 1.0)

        # Optional EMA smoothing across snapshots
        if i == WINDOW:
            topbottom_smooth = topbottom
        else:
            topbottom_smooth = (
                (1 - SMOOTH_EMA) * prev_tb + SMOOTH_EMA * topbottom
                if SMOOTH_EMA > 0
                else topbottom
            )
        prev_tb = topbottom_smooth

        ts = last_ts
        line = (
            f"[SNAPSHOT] {ts} | AvgPos:{avg_pos:.2f} | PosNow:{pos_now:.2f} | "
            f"TopBottom:{topbottom_smooth:.2f} | Skew:{wick_skew:+.02f} "
            f"| Wicks(uw={uw_ratio:.2f}, lw={lw_ratio:.2f})"
        )
        log_snapshot(line)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monthly wave snapshot with top/bottom score."
    )
    parser.add_argument("tag", help="Asset tag for CSV in data/raw/<TAG>.csv")
    args = parser.parse_args()
    run(args.tag)


if __name__ == "__main__":
    main()
