from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from typing import List, Tuple

# ts, open, high, low, close
Candle = Tuple[int, float, float, float, float]
LOG_PATH = Path("data/tmp/snapshots.log")

# Window and step sizes
WINDOW = 132
STEP = 15

# --- Top/Bottom score knobs ---
ALPHA_WICK = 0.12     # how strongly wicks skew PosNow toward 0/1
SMOOTH_EMA = 0.25     # 0 disables smoothing; else EMA factor in (0,1]
MOMENTUM_BARS = 8     # lookback for micro-slope (last N closes)
MOMENTUM_EPS = 0.0015
DEAD_ZONE_PCT = None  # override percentage; else derive from settings
DEAD_ZONE_MIN = 0.44  # fallback dead-zone bounds when pct not provided
DEAD_ZONE_MAX = 0.56

# SnapbackOdds knobs (structure-only; no new deps)
ODDS_LOOKBACK = 8        # bars for micro slope
W_DIVERGENCE  = 0.45     # weight: micro vs macro slope divergence
W_WICK        = 0.35     # weight: lower vs upper wick imbalance
W_DEPTH       = 0.20     # weight: depth below mid-tunnel


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


def run(tag: str, odds: bool = False) -> None:
    candles = load_candles(tag)
    if len(candles) < WINDOW:
        print("Not enough data to simulate.")
        return

    dz_pct = DEAD_ZONE_PCT
    if dz_pct is None:
        settings_path = Path("settings/settings.json")
        if settings_path.exists():
            try:
                settings = json.loads(settings_path.read_text())
                ledgers = settings.get("ledger_settings", {})
                ledger_iter = ledgers.values() if isinstance(ledgers, dict) else ledgers
                for ledger in ledger_iter:
                    if ledger.get("tag") == tag:
                        windows = ledger.get("window_settings", {})
                        for cfg in windows.values():
                            dz = cfg.get("dead_zone_pct")
                            if dz is not None:
                                dz_pct = dz
                                break
                        break
            except Exception:
                pass
    if dz_pct is not None:
        dead_zone_min = 0.5 - dz_pct / 2
        dead_zone_max = 0.5 + dz_pct / 2
    else:
        dead_zone_min = DEAD_ZONE_MIN
        dead_zone_max = DEAD_ZONE_MAX

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

        slope = (
            (closes[-1] - closes[-MOMENTUM_BARS]) / denom
            if len(closes) > MOMENTUM_BARS
            else 0.0
        )

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
        topbottom_base = clip(topbottom_raw + wick_skew, 0.0, 1.0)

        if dead_zone_min <= topbottom_base <= dead_zone_max:
            dead_zone_blend = 0.5 + 0.5 * (topbottom_base - 0.5)
            topbottom_filt = dead_zone_blend
        else:
            topbottom_filt = topbottom_base

        if topbottom_filt < dead_zone_min:
            if slope < -MOMENTUM_EPS:
                topbottom_filt = 0.75 * topbottom_filt + 0.25 * 0.5
        elif topbottom_filt > dead_zone_max:
            if slope > MOMENTUM_EPS:
                topbottom_filt = 0.75 * topbottom_filt + 0.25 * 0.5

        # ---- SnapbackOdds components ----
        # Wick imbalance (buyers rejecting lows)
        wick_balance = lw_ratio - uw_ratio

        # Micro vs. macro slope divergence
        if len(closes) > ODDS_LOOKBACK and denom > 1e-9:
            slope_micro = (closes[-1] - closes[-ODDS_LOOKBACK]) / denom
        else:
            slope_micro = 0.0
        slope_macro = (closes[-1] - closes[0]) / denom if denom > 1e-9 else 0.0
        macro_sign = 0.0 if slope_macro == 0 else (1.0 if slope_macro > 0 else -1.0)
        divergence = -slope_micro * macro_sign

        # Depth from mid-tunnel
        depth = (0.5 - pos_now) * 2.0

        raw = (
            W_DIVERGENCE * divergence
            + W_WICK * wick_balance
            + W_DEPTH * depth
        )
        snapback_up = clip(0.5 + 0.5 * raw, 0.0, 1.0)
        go_deeper = 1.0 - snapback_up

        # Optional EMA smoothing across snapshots
        if i == WINDOW:
            topbottom_smooth = topbottom_filt
        else:
            topbottom_smooth = (
                (1 - SMOOTH_EMA) * prev_tb + SMOOTH_EMA * topbottom_filt
                if SMOOTH_EMA > 0
                else topbottom_filt
            )
        prev_tb = topbottom_smooth

        ts = last_ts
        line = f"TopBottom:{topbottom_smooth:.2f}"
        if odds:
            line += f"  SnapbackOdds:{snapback_up:.2f}"
        log_snapshot(line)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monthly wave snapshot with top/bottom score."
    )
    parser.add_argument("tag", help="Asset tag for CSV in data/raw/<TAG>.csv")
    parser.add_argument(
        "--odds",
        action="store_true",
        help="Append SnapbackOdds to each TopBottom print",
    )
    args = parser.parse_args()
    run(args.tag, odds=args.odds)


if __name__ == "__main__":
    main()
