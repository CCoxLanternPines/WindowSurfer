from __future__ import annotations

import argparse
import re
from datetime import timedelta, datetime, timezone
import os

import pandas as pd
import numpy as np

from typing import Any, Dict, List

from systems.utils.settings_loader import load_coin_settings, load_general_settings
from systems.scripts.evaluate_buy import evaluate_buy
from systems.scripts.evaluate_sell import evaluate_sell
from systems.scripts.chart import plot_trades

# ===================== Parameters =====================
SIZE_SCALAR = 1_000_000
SIZE_POWER = 3


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

def apply_time_filter(
    df: pd.DataFrame, delta: timedelta, file_path: str, exhaustion_lookback: int
) -> pd.DataFrame:
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
    need = int(max(exhaustion_lookback * 2, delta.total_seconds() // sec))
    if need <= 0 or need >= len(df):
        return df
    return df.iloc[-need:]

# ===================== Exhaustion Plot + Trades =====================
def run_simulation(*, coin: str, timeframe: str = "1m", viz: bool = True) -> None:
    general_cfg = load_general_settings()
    coin_cfg = load_coin_settings(coin)

    start_capital = general_cfg.get("simulation_capital", 1000)
    monthly_topup = general_cfg.get("monthly_topup", 0)
    exhaustion_lookback = int(coin_cfg.get("exhaustion_lookback", 0))
    window_step = int(coin_cfg.get("window_step", 1))
    vol_lookback = int(coin_cfg.get("vol_lookback", 1))
    angle_lookback = int(coin_cfg.get("angle_lookback", 1))

    file_path = f"data/candles/sim/{coin}.csv"
    if not os.path.exists(file_path):
        file_path = f"data/sim/{coin}.csv"
    df = pd.read_csv(file_path)

    delta = parse_timeframe(timeframe)
    if delta is not None:
        df = apply_time_filter(df, delta, file_path, exhaustion_lookback)

    df = df.reset_index(drop=True)
    df["candle_index"] = range(len(df))

    df["returns"] = df["close"].pct_change()
    df["volatility"] = df["returns"].rolling(vol_lookback).std().fillna(0)

    # ----- Exhaustion points and rolling angles -----
    pts = {
        "exhaustion_up": {"x": [], "y": [], "s": []},
        "exhaustion_down": {"x": [], "y": [], "s": []},
    }
    df["angle"] = 0.0
    vol_pts = {"x": [], "y": [], "s": []}

    # Rolling slope calculation
    for t in range(angle_lookback, len(df)):
        dy = df["close"].iloc[t] - df["close"].iloc[t - angle_lookback]
        dx = angle_lookback
        angle = np.arctan2(dy, dx)
        norm = angle / (np.pi / 4)
        df.at[t, "angle"] = max(-1.0, min(1.0, norm))

    # Exhaustion bubbles
    for t in range(exhaustion_lookback, len(df), window_step):
        now_price = float(df["close"].iloc[t])
        past_price = float(df["close"].iloc[t - exhaustion_lookback])
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

    for t in range(vol_lookback, len(df), window_step):
        vol = df["volatility"].iloc[t]
        if pd.isna(vol):
            continue
        size = SIZE_SCALAR * (vol * 0.4 ** SIZE_POWER)
        vol_pts["x"].append(int(df["candle_index"].iloc[t]))
        vol_pts["y"].append(float(df["close"].iloc[t]))
        vol_pts["s"].append(size)

    # ===== Candle-by-candle simulation =====
    trades: List[Dict[str, Any]] = []
    capital = start_capital
    open_notes: List[Dict[str, float]] = []
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
                capital += monthly_topup
                last_month = current_month
                if monthly_topup:
                    print(
                        f"Monthly top-up: +{monthly_topup} USDT at {dt.date()} → Capital={capital:.2f}"
                    )

        price = row["close"]

        new_trades, capital, open_notes = evaluate_buy(
            idx, row, pts, capital, open_notes, coin_cfg
        )
        trades.extend(new_trades)

        closed_trades, capital, open_notes = evaluate_sell(
            idx, price, open_notes, capital, coin_cfg
        )
        trades.extend(closed_trades)

    # Final portfolio value
    final_value = capital + sum(n["units"] * float(df["close"].iloc[-1]) for n in open_notes)

    if viz:
        plot_trades(df, pts, vol_pts, trades, start_capital, final_value, coin_cfg)

    print(
        f"Final Capital: {capital:.2f}, Open Notes: {len(open_notes)}, Final Value: {final_value:.2f}"
    )

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--coin", required=True)
    p.add_argument("--time", type=str, default="1m")
    p.add_argument("--viz", action="store_true")
    args = p.parse_args()
    run_simulation(coin=args.coin, timeframe=args.time, viz=args.viz)


if __name__ == "__main__":
    main()
