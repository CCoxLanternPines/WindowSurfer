from __future__ import annotations

import argparse
import re
from datetime import timedelta, datetime, timezone
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ===================== Parameters =====================
WINDOW_SIZE      = 92
WINDOW_STEP      = 24
LOOKBACK         = 92
SIZE_SCALAR      = 1_000_000
SIZE_POWER       = 3

START_CAPITAL    = 10_000   # starting cash in USDT

# Buy scaling
BUY_MIN_BUBBLE    = 100
BUY_MAX_BUBBLE    = 500
MIN_NOTE_SIZE_PCT = 0.01    # 1% of portfolio
MAX_NOTE_SIZE_PCT = 0.05    # 5% of portfolio

# Sell scaling (baked into note at buy time)
SELL_MIN_BUBBLE   = 100
SELL_MAX_BUBBLE   = 800
MIN_MATURITY      = 0.03    # 0% gain (sell at entry)
MAX_MATURITY      = .3     # 100% gain (2x entry)

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

# ===================== Exhaustion Plot + Trades =====================
def run_simulation(*, timeframe: str = "1m", viz: bool = True) -> None:
    file_path = "data/sim/SOLUSD_1h.csv"
    df = pd.read_csv(file_path)

    delta = parse_timeframe(timeframe)
    if delta is not None:
        df = apply_time_filter(df, delta, file_path)

    df = df.reset_index(drop=True)
    df["candle_index"] = range(len(df))

    # ----- Exhaustion points -----
    pts = {
        "exhaustion_up":   {"x": [], "y": [], "s": []},
        "exhaustion_down": {"x": [], "y": [], "s": []},
    }

    for t in range(LOOKBACK, len(df), WINDOW_STEP):
        now_price = float(df["close"].iloc[t])
        past_price = float(df["close"].iloc[t - LOOKBACK])

        if now_price > past_price:
            delta_up = now_price - past_price
            norm = delta_up / max(1e-9, past_price)
            size = SIZE_SCALAR * (norm ** SIZE_POWER)
            pts["exhaustion_up"]["x"].append(int(df["candle_index"].iloc[t]))
            pts["exhaustion_up"]["y"].append(now_price)
            pts["exhaustion_up"]["s"].append(size)

        elif now_price < past_price:
            delta_down = past_price - now_price
            norm = delta_down / max(1e-9, past_price)
            size = SIZE_SCALAR * (norm ** SIZE_POWER)
            pts["exhaustion_down"]["x"].append(int(df["candle_index"].iloc[t]))
            pts["exhaustion_down"]["y"].append(now_price)
            pts["exhaustion_down"]["s"].append(size)

    # ===== Candle-by-candle simulation =====
    trades = []
    capital = START_CAPITAL
    open_notes = []

    for idx, row in df.iterrows():
        price = row["close"]

        # ---- Buy check ----
        if idx in pts["exhaustion_down"]["x"]:
            i = pts["exhaustion_down"]["x"].index(idx)
            bubble = pts["exhaustion_down"]["s"][i]
            total_cap = capital + sum(n["units"] * price for n in open_notes)
            trade_usd = scale_buy_size(bubble, total_cap)
            if trade_usd > 0 and capital >= trade_usd:
                units = trade_usd / price
                capital -= trade_usd
                sell_price = sell_target_from_bubble(price, bubble)
                open_notes.append({"entry_price": price, "units": units, "sell_price": sell_price})
                trades.append({"idx": idx, "price": price, "side": "BUY", "usd": trade_usd})
                print(f"BUY @ idx={idx}, price={price:.2f}, target={sell_price:.2f}")

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

        # Plot exhaustion bubbles
        ax1.scatter(pts["exhaustion_down"]["x"], pts["exhaustion_down"]["y"],
                    s=pts["exhaustion_down"]["s"], c="green", alpha=0.3, edgecolor="black")

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
