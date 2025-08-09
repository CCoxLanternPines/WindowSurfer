from __future__ import annotations

import argparse, csv, json
from pathlib import Path
from typing import List, Tuple

from systems.scripts.ledger_manager import LedgerManager
from systems.scripts.evaluate_buy import evaluate_buy
from systems.scripts.evaluate_sell import evaluate_sell

# ts, open, high, low, close
Candle = Tuple[int, float, float, float, float]
LOG_PATH = Path("data/tmp/snapshots.log")

# Window and step sizes
WINDOW = 72
STEP = 5

# --- Top/Bottom score knobs ---
ALPHA_WICK = 0.12     # how strongly wicks skew PosNow toward 0/1
SMOOTH_EMA = 0.25     # 0 disables smoothing; else EMA factor in (0,1]
MOMENTUM_BARS = 8     # lookback for micro-slope (last N closes)
MOMENTUM_EPS = 0.0015
DEAD_ZONE_PCT = None  # override percentage; else derive from settings
DEAD_ZONE_MIN = 0.44  # fallback dead-zone bounds when pct not provided
DEAD_ZONE_MAX = 0.56

BASE_UNIT = 1.0

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


def run(tag: str) -> None:
    candles = load_candles(tag)
    if len(candles) < WINDOW:
        print("Not enough data to simulate.")
        return

    # Resolve dead-zone from settings (optional)
    dz_pct = DEAD_ZONE_PCT
    if dz_pct is None:
        settings_path = Path("settings/settings.json")
        if settings_path.exists():
            try:
                settings = json.loads(settings_path.read_text())
                for ledger in (settings.get("ledger_settings") or {}).values():
                    if ledger.get("tag") == tag:
                        for cfg in (ledger.get("window_settings") or {}).values():
                            dz = cfg.get("dead_zone_pct")
                            if dz is not None:
                                dz_pct = dz
                                break
                        break
            except Exception:
                pass
    dead_zone_min = 0.5 - dz_pct / 2 if dz_pct is not None else DEAD_ZONE_MIN
    dead_zone_max = 0.5 + dz_pct / 2 if dz_pct is not None else DEAD_ZONE_MAX

    init_snapshot_log()
    prev_tb = 0.5

    ledger = LedgerManager(tag)


    for i in range(WINDOW, len(candles), STEP):
        window = candles[i - WINDOW : i]
        lows   = [x[3] for x in window]
        highs  = [x[2] for x in window]
        closes = [x[4] for x in window]
        price = closes[-1]
        ts = window[-1][0]

        low_w, high_w = min(lows), max(highs)
        denom = max(1e-9, high_w - low_w)

        # Micro slope
        slope = (closes[-1] - closes[-MOMENTUM_BARS]) / denom if len(closes) > MOMENTUM_BARS else 0.0

        # Window position
        pos_now = 0.5 if high_w == low_w else (closes[-1] - low_w) / denom
        pos_now = clip(pos_now, 0.0, 1.0)

        # Wick features (last candle)
        _, o, h, l, c = window[-1]
        rng = max(1e-9, h - l)
        body_low, body_high = (min(o, c), max(o, c))
        lower_wick = body_low - l
        upper_wick = h - body_high
        lw_ratio = clip(lower_wick / rng, 0.0, 1.0)
        uw_ratio = clip(upper_wick / rng, 0.0, 1.0)

        # TopBottom (with wick skew + dead-zone filter + momentum confirm)
        wick_skew = ALPHA_WICK * (uw_ratio - lw_ratio)
        topbottom_base = clip(pos_now + wick_skew, 0.0, 1.0)

        if dead_zone_min <= topbottom_base <= dead_zone_max:
            topbottom_filt = 0.5 + 0.5 * (topbottom_base - 0.5)  # halve deviation
        else:
            topbottom_filt = topbottom_base

        if topbottom_filt < dead_zone_min:
            if slope < -MOMENTUM_EPS:
                topbottom_filt = 0.75 * topbottom_filt + 0.25 * 0.5
        elif topbottom_filt > dead_zone_max:
            if slope > MOMENTUM_EPS:
                topbottom_filt = 0.75 * topbottom_filt + 0.25 * 0.5

        # Snapback (structure-only): divergence + wick balance + depth
        wick_balance = lw_ratio - uw_ratio
        if len(closes) > ODDS_LOOKBACK and denom > 1e-9:
            slope_micro = (closes[-1] - closes[-ODDS_LOOKBACK]) / denom
        else:
            slope_micro = 0.0

        slope_macro = (closes[-1] - closes[0]) / denom if denom > 1e-9 else 0.0
        macro_sign = 0.0 if slope_macro == 0 else (1.0 if slope_macro > 0 else -1.0)
        divergence = -slope_micro * macro_sign  # positive when micro fights macro

        depth = (0.5 - pos_now) * 2.0  # floor:+1, ceiling:-1

        raw = (W_DIVERGENCE * divergence) + (W_WICK * wick_balance) + (W_DEPTH * depth)
        snapback_up = clip(0.5 + 0.5 * raw, 0.0, 1.0)

        # Directional ramps (your request):
        # SellVar: 0.5 -> 0, 0.3 -> 1
        sell_var = (0.5 - snapback_up) / 0.2
        sell_var = clip(sell_var, 0.0, 1.0)
        # BuyVar: 0.5 -> 0, 0.7 -> 1
        buy_var = (snapback_up - 0.5) / 0.2
        buy_var = clip(buy_var, 0.0, 1.0)

        # EMA smoothing for TopBottom
        if i == WINDOW:
            topbottom_smooth = topbottom_filt
        else:
            topbottom_smooth = (1 - SMOOTH_EMA) * prev_tb + SMOOTH_EMA * topbottom_filt if SMOOTH_EMA > 0 else topbottom_filt
        prev_tb = topbottom_smooth

        # Directional ramps (your request):
        # SellVar: 0.5 -> 0, 0.3 -> 1
        sell_var = (0.5 - snapback_up) / 0.2
        sell_var = clip(sell_var, 0.0, 1.0)
        # BuyVar: 0.5 -> 0, 0.7 -> 1
        buy_var = (snapback_up - 0.5) / 0.2
        buy_var = clip(buy_var, 0.0, 1.0)

        # keep your existing sell_var / buy_var block here

        # 75/25 weighting: TopBottom dominates, Var tops it off
        should_sell = 0.75 * topbottom_smooth + 0.25 * sell_var
        should_buy  = 0.75 * (1 - topbottom_smooth) + 0.25 * buy_var

        # Remap: 0.5 → 0, 1 → 1, values below 0.5 become negative unless clipped
        should_sell = 2 * (should_sell - 0.5) / 0.5
        should_sell = clip(should_sell, 0.0, 1.0)

        should_buy  = 2 * (should_buy - 0.5) / 0.5
        should_buy  = clip(should_buy, 0.0, 1.0)

        ctx = {
            "topbottom": topbottom_smooth,
            "buy_var": buy_var,
            "sell_var": sell_var,
            "should_buy": should_buy,
            "should_sell": should_sell,
            "base_unit": BASE_UNIT,
            "price": price,
            "ts": ts,
            "total_coin": ledger.total_coin(),
        }

        buy_amt = evaluate_buy(ctx)
        sell_amt = evaluate_sell(ctx)

        action = None
        if sell_amt > 0:
            ledger.sell(sell_amt, ctx["price"], ctx["ts"])
            action = f"SELLx{sell_amt:.2f}"
        elif buy_amt > 0:
            ledger.buy(buy_amt, ctx["price"], ctx["ts"])
            action = f"BUYx{buy_amt:.2f}"

        # Single clean print
        suffix = ""
        if action:
            suffix = f" Action:{action}"

        line = (
            f"TopBottom:{topbottom_smooth:.2f} "
            f"SellVar:{sell_var:.2f} BuyVar:{buy_var:.2f} "
            f"ShouldBuy:{should_buy:.2f} ShouldSell:{should_sell:.2f}{suffix}"
        )
        line += f" | OpenNotes:{len(ledger.get_open_notes())} Coin:{ledger.total_coin():.4f}"
        log_snapshot(line)

    final_price = candles[-1][4]
    realized = ledger.realized_gain_usd()
    unrealized = ledger.unrealized_gain_usd(final_price)
    total = realized + unrealized
    coin = ledger.total_coin()
    print(
        f"[SIM] PnL | Realized:${realized:.2f} | Unrealized:${unrealized:.2f} | "
        f"Total:${total:.2f} | Coin:{coin:.6f} @ ${final_price:.4f}"
    )

    out_dir = Path("data/tmp")
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger.save(out_dir / "ledger_simple.json")
    ledger.save_summary(Path("data/tmp/ledger_simple_summary.json"), final_price)


def main() -> None:
    parser = argparse.ArgumentParser(description="Monthly wave snapshot with top/bottom score.")
    parser.add_argument("tag", help="Asset tag for CSV in data/raw/<TAG>.csv")
    args = parser.parse_args()
    run(args.tag)


if __name__ == "__main__":
    main()
