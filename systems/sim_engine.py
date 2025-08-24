from __future__ import annotations

import argparse
import re
from datetime import timedelta, datetime, timezone
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ===================== Parameters =====================
# Lookbacks

SIZE_SCALAR      = 1_000_000
SIZE_POWER       = 3

START_CAPITAL    = 10_000   # starting cash in USDT
MONTHLY_TOPUP    = 000    # fixed USDT injected each calendar month

EXHAUSTION_LOOKBACK = 184   # used for bubble delta
WINDOW_STEP = 12

# Buy scaling
BUY_MIN_BUBBLE    = 100
BUY_MAX_BUBBLE    = 500
MIN_NOTE_SIZE_PCT = 0.03    # 1% of portfolio
MAX_NOTE_SIZE_PCT = 0.2    # 5% of portfolio

# Sell scaling (baked into note at buy time)
SELL_MIN_BUBBLE   = 100
SELL_MAX_BUBBLE   = 800
MIN_MATURITY      = 0.03    # 0% gain (sell at entry)
MAX_MATURITY      = .3     # 100% gain (2x entry)

# Trend multipliers
BUY_MULT_TREND_UP   = 1   # strong up-trend multiplier (cap at +1 normalized)
BUY_MULT_TREND_FLOOR = .25  # keep 0 so flat maps to 0, no forced minimum
BUY_MULT_TREND_DOWN = 0   # strong down-trend multiplier (cap at -1 normalized)

# Volatility buy scaling
BUY_MIN_VOL_BUBBLE = 0
BUY_MAX_VOL_BUBBLE = 500
BUY_MULT_VOL_MIN   = 0.25
BUY_MULT_VOL_MAX   = 1.0

VOL_LOOKBACK = 48   # rolling window for volatility

# Angle thresholds (normalized; 0.0..1.0 where 1.0 = 45°)
ANGLE_UP_MIN   = 0.01   # require at least +0.20 (~+9°) to start scaling up
ANGLE_DOWN_MIN = 0.50   # require at least -0.20 (~-9°) to start scaling down
ANGLE_LOOKBACK = 48     # used for slope angle

_INTERVAL_RE = re.compile(r'[_\-]((\d+)([smhdw]))(?=\.|_|$)', re.I)

TIMEFRAME_SECONDS = {
    's': 1,
    'm': 30 * 24 * 3600,
    'h': 3600,
    'd': 86400,
    'w': 604800,
}

INTERVAL_SECONDS = {
    's': 1,
    'm': 60,
    'h': 3600,
    'd': 86400,
    'w': 604800,
}

def parse_timeframe(tf: str):
    if not tf:
        return None
    m = re.match(r'(?i)^\s*(\d+)\s*([smhdw])\s*$', tf)
    if not m:
        return None
    n, u = int(m.group(1)), m.group(2).lower()
    return timedelta(seconds=n * TIMEFRAME_SECONDS[u])

def infer_candle_seconds_from_filename(path: str) -> int | None:
    m = _INTERVAL_RE.search(os.path.basename(path))
    if not m:
        return None
    n, u = int(m.group(2)), m.group(3).lower()
    return n * INTERVAL_SECONDS[u]

def apply_time_filter(df: pd.DataFrame, delta: timedelta, file_path: str) -> pd.DataFrame:
    if delta is None:
        return df
    if 'timestamp' in df.columns:
        ts = df['timestamp']
        ts_max = float(ts.iloc[-1])
        is_ms = ts_max > 1e12
        to_seconds = (ts / 1000.0) if is_ms else ts
        cutoff = (datetime.now(timezone.utc).timestamp() - delta.total_seconds())
        mask = to_seconds >= cutoff
        return df.loc[mask]
    for col in ('datetime','date','time'):
        if col in df.columns:
            try:
                dt = pd.to_datetime(df[col], utc=True, errors='coerce')
                cutoff_dt = pd.Timestamp.utcnow() - delta
                mask = dt >= cutoff_dt
                return df.loc[mask]
            except Exception:
                pass
    sec = infer_candle_seconds_from_filename(file_path) or 3600
    need = int(max(WINDOW_SIZE, delta.total_seconds() // sec))
    if need <= 0 or need >= len(df):
        return df
    return df.iloc[-need:]

# ===================== Trade sizing =====================
def scale_buy_size(s: float, total_cap: float) -> float:
    if s < BUY_MIN_BUBBLE:
        return 0.0
    s_clamped = min(max(s, BUY_MIN_BUBBLE), BUY_MAX_BUBBLE)
    frac = (s_clamped - BUY_MIN_BUBBLE) / (BUY_MAX_BUBBLE - BUY_MIN_BUBBLE)
    pct = MIN_NOTE_SIZE_PCT + frac * (MAX_NOTE_SIZE_PCT - MIN_NOTE_SIZE_PCT)
    return total_cap * pct

def sell_target_from_bubble(entry_price: float, s: float) -> float:
    if s < SELL_MIN_BUBBLE:
        return float("inf")  # effectively never sell
    s_clamped = min(max(s, SELL_MIN_BUBBLE), SELL_MAX_BUBBLE)
    frac = (s_clamped - SELL_MIN_BUBBLE) / (SELL_MAX_BUBBLE - SELL_MIN_BUBBLE)
    maturity = MIN_MATURITY + frac * (MAX_MATURITY - MIN_MATURITY)
    return entry_price * (1 + maturity)
def trend_multiplier_lerp(v: float) -> float:
    """
    Linear interpolation of multiplier from angle:
      v=-1 → BUY_MULT_TREND_DOWN
      v= 0 → BUY_MULT_TREND_FLOOR
      v=+1 → BUY_MULT_TREND_UP
    """
    v = max(-1.0, min(1.0, float(v)))
    if v < 0:
        # interpolate between -1 and 0
        return BUY_MULT_TREND_DOWN + (BUY_MULT_TREND_FLOOR - BUY_MULT_TREND_DOWN) * (v + 1)
    else:
        # interpolate between 0 and +1
        return BUY_MULT_TREND_FLOOR + (BUY_MULT_TREND_UP - BUY_MULT_TREND_FLOOR) * v


def vol_multiplier(vol: float) -> float:
    """
    Map rolling volatility into buy multiplier.
    vol → scaled between BUY_MIN_VOL_BUBBLE and BUY_MAX_VOL_BUBBLE.
    """
    # clamp to bubble range
    v = min(max(vol, BUY_MIN_VOL_BUBBLE), BUY_MAX_VOL_BUBBLE)
    frac = (v - BUY_MIN_VOL_BUBBLE) / max(1e-9, (BUY_MAX_VOL_BUBBLE - BUY_MIN_VOL_BUBBLE))
    return BUY_MULT_VOL_MIN + frac * (BUY_MULT_VOL_MAX - BUY_MULT_VOL_MIN)

# ===================== Exhaustion Plot + Trades =====================
def run_simulation(*, timeframe: str = "1m", viz: bool = True) -> None:
    file_path = "data/sim/SOLUSD_1h.csv"
    df = pd.read_csv(file_path)

    delta = parse_timeframe(timeframe)
    if delta is not None:
        df = apply_time_filter(df, delta, file_path)

    df = df.reset_index(drop=True)
    df["candle_index"] = range(len(df))

    df["returns"] = df["close"].pct_change()
    df["volatility"] = df["returns"].rolling(VOL_LOOKBACK).std().fillna(0)

    # ----- Exhaustion points and rolling angles -----
    pts = {
        "exhaustion_up":   {"x": [], "y": [], "s": []},
        "exhaustion_down": {"x": [], "y": [], "s": []},
    }
    df["angle"] = 0.0
    vol_pts = {"x": [], "y": [], "s": []}

    # Rolling slope calculation
    for t in range(ANGLE_LOOKBACK, len(df)):
        dy = df["close"].iloc[t] - df["close"].iloc[t - ANGLE_LOOKBACK]
        dx = ANGLE_LOOKBACK
        angle = np.arctan2(dy, dx)
        norm = angle / (np.pi / 4)
        df.at[t, "angle"] = max(-1.0, min(1.0, norm))

    # Exhaustion bubbles
    for t in range(EXHAUSTION_LOOKBACK, len(df), WINDOW_STEP):
        now_price = float(df["close"].iloc[t])
        past_price = float(df["close"].iloc[t - EXHAUSTION_LOOKBACK])
        end_idx = int(df["candle_index"].iloc[t])

        if now_price > past_price:
            delta_up = now_price - past_price
            norm_up = delta_up / max(1e-9, past_price)
            size = SIZE_SCALAR * (norm_up ** SIZE_POWER)
            pts["exhaustion_up"]["x"].append(end_idx)
            pts["exhaustion_up"]["y"].append(now_price)
            pts["exhaustion_up"]["s"].append(size)

        elif now_price < past_price:
            delta_down = past_price - now_price
            norm_down = delta_down / max(1e-9, past_price)
            size = SIZE_SCALAR * (norm_down ** SIZE_POWER)
            pts["exhaustion_down"]["x"].append(end_idx)
            pts["exhaustion_down"]["y"].append(now_price)
            pts["exhaustion_down"]["s"].append(size)

    for t in range(VOL_LOOKBACK, len(df), WINDOW_STEP):
        vol = df["volatility"].iloc[t]
        if pd.isna(vol):
            continue
        size = SIZE_SCALAR * (vol ** SIZE_POWER)
        vol_pts["x"].append(int(df["candle_index"].iloc[t]))
        vol_pts["y"].append(float(df["close"].iloc[t]))
        vol_pts["s"].append(size)

    # ===== Candle-by-candle simulation =====
    trades = []
    capital = START_CAPITAL
    open_notes = []
    last_month = None

    for idx, row in df.iterrows():
        dt = None
        if 'timestamp' in row:
            ts = float(row['timestamp'])
            is_ms = ts > 1e12
            dt = datetime.fromtimestamp(ts / (1000 if is_ms else 1), tz=timezone.utc)
        elif 'datetime' in row:
            try:
                dt = pd.to_datetime(row['datetime'], utc=True).to_pydatetime()
            except Exception:
                dt = None
        if dt is not None:
            current_month = (dt.year, dt.month)
            if current_month != last_month:
                capital += MONTHLY_TOPUP
                last_month = current_month
                print(f"Monthly top-up: +{MONTHLY_TOPUP} USDT at {dt.date()} → Capital={capital:.2f}")

        price = row["close"]

        # ---- Buy check ----
        if idx in pts["exhaustion_down"]["x"]:
            i = pts["exhaustion_down"]["x"].index(idx)
            bubble = pts["exhaustion_down"]["s"][i]
            total_cap = capital + sum(n["units"] * price for n in open_notes)
            trade_usd = scale_buy_size(bubble, total_cap)

            # angle-based multiplier
            v = row["angle"]
            trend_mult = trend_multiplier_lerp(v)

            # volatility-based multiplier
            vol = row["volatility"]
            vol_mult = vol_multiplier(vol)

            # combine
            trade_usd *= max(BUY_MULT_TREND_FLOOR, trend_mult)
            trade_usd *= vol_mult
            if trade_usd > 0 and capital >= trade_usd:
                units = trade_usd / price
                capital -= trade_usd
                sell_price = sell_target_from_bubble(price, bubble)
                open_notes.append({"entry_price": price, "units": units, "sell_price": sell_price})
                trades.append({"idx": idx, "price": price, "side": "BUY", "usd": trade_usd})
                print(
                    f"BUY @ idx={idx}, price={price:.2f}, angle_mult={trend_mult:.2f}, vol_mult={vol_mult:.2f}, target={sell_price:.2f}"
                )

        # ---- Sell check ----
        closed_notes = []
        for note in open_notes:
            if price >= note["sell_price"]:
                sell_usd = note["units"] * price
                capital += sell_usd
                trades.append({"idx": idx, "price": price, "side": "SELL", "usd": sell_usd})
                closed_notes.append(note)
                print(f"SELL @ idx={idx}, entry={note['entry_price']:.2f}, target={note['sell_price']:.2f}, price={price:.2f}")
        for n in closed_notes:
            open_notes.remove(n)

    # Final portfolio value
    final_value = capital + sum(n["units"] * float(df["close"].iloc[-1]) for n in open_notes)

    if viz:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(df["candle_index"], df["close"], lw=1, label="Close Price", color="blue")
        ax1.set_title(f"Exhaustion Trades\nStart {START_CAPITAL}, End {final_value:.2f}")
        ax1.set_xlabel("Candles (Index)")
        ax1.set_ylabel("Price")
        ax1.grid(True)

        # Plot rolling slope arrows
        for i, r in df.iterrows():
            v = r["angle"]
            if i < ANGLE_LOOKBACK:
                continue
            if v > 0.05:
                color = "orange"
            elif v < -0.05:
                color = "purple"
            else:
                color = "gray"
            x0, y0 = r["candle_index"], r["close"]
            x1 = x0 + 5
            y1 = y0 + v * 5
            ax1.plot([x0, x1], [y0, y1], color=color, lw=1.5, alpha=0.7)

        # Plot exhaustion bubbles
        ax1.scatter(pts["exhaustion_down"]["x"], pts["exhaustion_down"]["y"],
                    s=pts["exhaustion_down"]["s"], c="green", alpha=0.3, edgecolor="black")

        # Plot volatility bubbles (red)
        ax1.scatter(vol_pts["x"], vol_pts["y"],
                    s=vol_pts["s"], c="red", alpha=0.3, edgecolor="black")

        # Plot trades
        for t in trades:
            if t["side"] == "BUY":
                ax1.scatter(t["idx"], t["price"], marker="^", s=150,
                            c="lime", edgecolor="black", zorder=10)
            elif t["side"] == "SELL":
                ax1.scatter(t["idx"], t["price"], marker="v", s=150,
                            c="red", edgecolor="black", zorder=10)

        plt.show()

    print(f"Final Capital: {capital:.2f}, Open Notes: {len(open_notes)}, Final Value: {final_value:.2f}")

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--time", type=str, default="1m")
    p.add_argument("--viz", action="store_true")
    args = p.parse_args()
    run_simulation(timeframe=args.time, viz=args.viz)

if __name__ == "__main__":
    main()
