from __future__ import annotations
import argparse, csv, math
from pathlib import Path
from typing import List, Tuple, Optional

# ts, open, high, low, close
Candle = Tuple[int, float, float, float, float]
LOG_PATH = Path("data/tmp/snapshots.log")

# Tunable constants
WINDOW = 132
STEP = 24
K = 12
H = 48
UP_TAKE = 0.03
DOWN_FAIL = -0.015
W_NEAR = 0.45
W_MOMO = 0.25
W_WICK = 0.20
W_HL = 0.10


def clip(x: float, a: float, b: float) -> float:
    return max(a, min(b, x))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def compute_bottom_prob(window: List[Candle], pos_now: float) -> Tuple[float, float, float, float, float]:
    closes = [x[4] for x in window]
    lows = [x[3] for x in window]
    opens = [x[1] for x in window]
    highs = [x[2] for x in window]

    f_near = 1 - pos_now

    ma_short = sum(closes[-8:]) / max(1e-9, len(closes[-8:]))
    ma_long = sum(closes[-24:]) / max(1e-9, len(closes[-24:]))
    momo_raw = (ma_short - ma_long) / max(1e-9, ma_long)
    f_momo = sigmoid(momo_raw * 10)

    open_, high, low, close = opens[-1], highs[-1], lows[-1], closes[-1]
    lower_wick = min(open_, close) - low
    range_ = high - low
    f_wick = clip(lower_wick / max(1e-9, range_), 0.0, 1.0)

    k_lows = lows[-K:]
    if len(k_lows) < 2:
        f_hl = 0.0
    else:
        prior_min = min(k_lows[:-1])
        f_hl = 1.0 if k_lows[-1] > prior_min * (1 + 0.001) else 0.0

    bottom_prob = clip(
        W_NEAR * f_near + W_MOMO * f_momo + W_WICK * f_wick + W_HL * f_hl,
        0.0,
        1.0,
    )

    return bottom_prob, f_near, f_momo, f_wick, f_hl


def compute_ground_truth(candles: List[Candle], idx: int) -> Optional[int]:
    if idx + H >= len(candles):
        return None

    t0_close = candles[idx - 1][4]
    t0_low = candles[idx - 1][3]
    up_target = t0_close * (1 + UP_TAKE)
    down_target = t0_low * (1 + DOWN_FAIL)

    for j in range(idx, idx + H):
        high = candles[j][2]
        low = candles[j][3]
        if high >= up_target:
            return 1
        if low <= down_target:
            return 0
    return 0

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
    if len(candles) < WINDOW + H:
        print("Not enough data to simulate.")
        return

    init_snapshot_log()

    bottom_probs: List[float] = []
    gts: List[int] = []

    for i in range(WINDOW, len(candles), STEP):
        window = candles[i - WINDOW : i]
        lows = [x[3] for x in window]
        highs = [x[2] for x in window]
        closes = [x[4] for x in window]

        low_w = min(lows)
        high_w = max(highs)

        current_close = closes[-1]
        denom = max(1e-9, high_w - low_w)
        pos_now = 0.5 if high_w == low_w else (current_close - low_w) / denom
        pos_now = clip(pos_now, 0.0, 1.0)

        bottom_prob, f_near, f_momo, f_wick, f_hl = compute_bottom_prob(window, pos_now)

        gt = compute_ground_truth(candles, i)
        if gt is not None:
            bottom_probs.append(bottom_prob)
            gts.append(gt)

        ts = window[-1][0]
        gt_str = str(gt) if gt is not None else "NA"
        debug_bits = (
            f"near={f_near:.2f}, momo={f_momo:.2f}, "
            f"wick={f_wick:.2f}, hl={int(f_hl)}"
        )
        line = (
            f"[SNAPSHOT] {ts} | PosNow:{pos_now:.2f} | BottomProb:{bottom_prob:.2f} "
            f"| GT:{gt_str} | Notes:{debug_bits}"
        )
        log_snapshot(line)

    # End-of-run summary
    if gts:
        gt1_probs = [p for p, g in zip(bottom_probs, gts) if g == 1]
        gt0_probs = [p for p, g in zip(bottom_probs, gts) if g == 0]
        avg_gt1 = sum(gt1_probs) / len(gt1_probs) if gt1_probs else 0.0
        avg_gt0 = sum(gt0_probs) / len(gt0_probs) if gt0_probs else 0.0

        high_cases = [(p, g) for p, g in zip(bottom_probs, gts) if p >= 0.7]
        hits_high = sum(1 for p, g in high_cases if g == 1)
        count_high = len(high_cases)
        hit_rate_high = hits_high / count_high if count_high else 0.0

        summary = (
            f"[SUMMARY] AvgProb|GT=1:{avg_gt1:.3f} AvgProb|GT=0:{avg_gt0:.3f} "
            f"HitRate>=0.7:{hit_rate_high:.3f}"
        )
        log_snapshot(summary)

        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        for start, end in zip(bins[:-1], bins[1:]):
            probs = [g for p, g in zip(bottom_probs, gts) if start <= p < end]
            rate = sum(probs) / len(probs) if probs else 0.0
            log_snapshot(
                f"[CALIBRATION] {start:.1f}-{end:.1f}: {rate:.3f}",
                also_print=False,
            )

def main():
    parser = argparse.ArgumentParser(description="Monthly wave snapshot with predictive tags.")
    parser.add_argument("tag", help="Asset tag for CSV in data/raw/<TAG>.csv")
    args = parser.parse_args()
    run(args.tag)

if __name__ == "__main__":
    main()
