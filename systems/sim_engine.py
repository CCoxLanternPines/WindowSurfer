from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np

from systems.scripts.evaluate_buy import evaluate_buy
from systems.scripts.evaluate_sell import evaluate_sell
from systems.utils.time import parse_timeframe, apply_time_filter
from systems.utils import log
from systems.utils.graph_feed import GraphFeed

# ===================== Parameters =====================
# Lookbacks

SIZE_SCALAR      = 1_000_000
SIZE_POWER       = 3

START_CAPITAL    = 10_000   # starting cash in USDT
MONTHLY_TOPUP    = 000    # fixed USDT injected each calendar month

EXHAUSTION_LOOKBACK = 184   # used for bubble delta
WINDOW_STEP = 12

VOL_LOOKBACK = 48   # rolling window for volatility
ANGLE_LOOKBACK = 48     # used for slope angle



# ===================== Exhaustion Plot + Trades =====================
def run_simulation(
    *,
    coin: str,
    timeframe: str = "1m",
    graph_feed: bool = True,
    graph_downsample: int = 1,
    viz: bool = False,
) -> None:
    """Run historical simulation for ``coin``.

    Parameters
    ----------
    coin:
        Market symbol (e.g. ``DOGEUSD``).
    timeframe:
        Optional timeframe filter. Defaults to "1m".
    viz:
        Deprecated. Plotting handled by systems/graph_engine.py.
    """

    coin = coin.replace("/", "").upper()
    file_path = f"data/sim/{coin}.csv"
    try_alt = f"data/candles/sim/{coin}.csv"
    import os, pandas as pd
    if not os.path.exists(file_path) and os.path.exists(try_alt):
        file_path = try_alt
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
    vol_pts = {"x": [], "y": [], "s": []}
    df["angle"] = 0.0

    feed = GraphFeed(mode="sim", coin=coin, downsample=graph_downsample, flush=False) if graph_feed else None

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
    # Volatility bubbles
    for t in range(VOL_LOOKBACK, len(df), WINDOW_STEP):
        price = float(df["close"].iloc[t])
        vol = float(df["volatility"].iloc[t])
        end_idx = int(df["candle_index"].iloc[t])
        size = SIZE_SCALAR * (vol ** SIZE_POWER)
        vol_pts["x"].append(end_idx)
        vol_pts["y"].append(price)
        vol_pts["s"].append(size)

    if feed:
        # pressure/green bubbles
        for x, y, s in zip(
            pts["exhaustion_down"]["x"],
            pts["exhaustion_down"]["y"],
            pts["exhaustion_down"]["s"],
        ):
            feed.pressure_bubble(int(x), float(y), float(s))
        # volatility/red bubbles
        for x, y, s in zip(vol_pts["x"], vol_pts["y"], vol_pts["s"]):
            feed.vol_bubble(int(x), float(y), float(s))

    # ===== Candle-by-candle simulation =====
    trades = []
    capital = START_CAPITAL
    open_notes = []
    last_month = None

    for idx, row in df.iterrows():
        dt = None
        ts_val = None
        if "timestamp" in row:
            ts = float(row["timestamp"])
            is_ms = ts > 1e12
            ts_val = int(ts / (1000 if is_ms else 1))
            dt = datetime.fromtimestamp(ts / (1000 if is_ms else 1), tz=timezone.utc)
        elif "datetime" in row:
            try:
                dt = pd.to_datetime(row["datetime"], utc=True).to_pydatetime()
            except Exception:
                dt = None
        if dt is not None:
            current_month = (dt.year, dt.month)
            if current_month != last_month:
                capital += MONTHLY_TOPUP
                last_month = current_month
                log.what(
                    f"Monthly top-up: +{MONTHLY_TOPUP} USDT at {dt.date()} → Capital={capital:.2f}"
                )

        price = row["close"]

        if feed:
            feed.candle(
                int(row["candle_index"]),
                ts_val,
                float(row.get("open", price)),
                float(row.get("high", price)),
                float(row.get("low", price)),
                float(price),
            )
            if idx >= ANGLE_LOOKBACK and idx % WINDOW_STEP == 0:
                feed.indicator(int(row["candle_index"]), "angle", float(row.get("angle", 0.0)))
            if idx >= VOL_LOOKBACK and idx % WINDOW_STEP == 0:
                feed.indicator(int(row["candle_index"]), "vol", float(row.get("volatility", 0.0)))

        prev_len = len(open_notes)
        trade, capital, open_notes = evaluate_buy(idx, row, pts, capital, open_notes)
        if trade:
            trades.append(trade)
            if feed and len(open_notes) > prev_len:
                note = open_notes[-1]
                feed.buy(
                    trade["idx"],
                    trade["price"],
                    float(note.get("units", 0.0)),
                    trade["usd"],
                    float(note.get("sell_price", 0.0)),
                )

        prev_notes = list(open_notes)
        closed, capital, open_notes = evaluate_sell(idx, price, open_notes, capital)
        trades.extend(closed)
        if feed and closed:
            closed_notes = [n for n in prev_notes if n not in open_notes]
            for tr, n in zip(closed, closed_notes):
                feed.sell(
                    tr["idx"],
                    tr["price"],
                    float(n.get("units", 0.0)),
                    tr["usd"],
                    float(n.get("entry_price", 0.0)),
                )

        if feed and (idx % 50 == 0):
            equity = capital + sum(n["units"] * price for n in open_notes)
            feed.capital(int(row["candle_index"]), float(capital), float(equity))

    # Final portfolio value
    final_value = capital + sum(n["units"] * float(df["close"].iloc[-1]) for n in open_notes)

    if viz:
        pass  # plotting handled by systems/graph_engine.py

    if feed:
        feed.close()

    log.what(
        f"Final Capital: {capital:.2f}, Open Notes: {len(open_notes)}, Final Value: {final_value:.2f}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--coin", required=True)
    p.add_argument("--time", type=str, default="1m")
    p.add_argument("--graph-feed", action="store_true", default=True, help="Emit NDJSON graph feed for graph_engine")
    p.add_argument("--graph-downsample", type=int, default=1, help="Downsample factor for feed")
    p.add_argument("-v", action="count", default=0, help="Increase verbosity (use -vv for more)")
    p.add_argument("--log", action="store_true", help="Write logs to file")
    args = p.parse_args()

    coin = args.coin.replace("/", "").upper()
    log.init_logger(verbosity=1 + args.v, to_file=args.log, name_hint=f"sim_{coin}")
    log.what(f"Running simulation for {coin} with timeframe {args.time}")
    run_simulation(
        coin=coin,
        timeframe=args.time,
        graph_feed=args.graph_feed,
        graph_downsample=args.graph_downsample,
        viz=False,
    )


if __name__ == "__main__":
    main()
