from __future__ import annotations

import argparse
import json
import pathlib
import numpy as np

import systems.scripts.evaluate_buy as buy_module
from systems.scripts.evaluate_sell import evaluate_sell
from systems.utils.time import parse_timeframe, apply_time_filter
from systems.utils import log
from systems.utils.graph_feed import GraphFeed
from systems.utils.config_loader import load_coin_settings
from systems.scripts.candle_loop import run_candle_loop


SETTINGS_PATH = pathlib.Path("settings/settings.json")

if SETTINGS_PATH.exists():
    _settings = json.loads(SETTINGS_PATH.read_text())
    _general = _settings.get("general_settings", {})
    START_CAPITAL = _general.get("simulation_capital", 10_000)
    MONTHLY_TOPUP = _general.get("monthly_topup", 0)
else:
    START_CAPITAL = 10_000
    MONTHLY_TOPUP = 0

# ===================== Parameters =====================
# Lookbacks and scalars
# ===================== Parameters =====================
# Lookbacks and scalars

SIZE_SCALAR      = 1_000_000
SIZE_POWER       = 3

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

    # Load per-coin settings with safe fallbacks
    settings = load_coin_settings(coin)
    EXHAUSTION_LOOKBACK = int(settings.get("exhaustion_lookback", 184))
    WINDOW_STEP = int(settings.get("window_step", 12))

    BUY_MIN_BUBBLE = float(settings.get("buy_min_bubble", 100))
    BUY_MAX_BUBBLE = float(settings.get("buy_max_bubble", 500))
    MIN_NOTE_SIZE_PCT = float(settings.get("min_note_size_pct", 0.03))
    MAX_NOTE_SIZE_PCT = float(settings.get("max_note_size_pct", 0.25))

    SELL_MIN_BUBBLE = float(settings.get("sell_min_bubble", 150))
    SELL_MAX_BUBBLE = float(settings.get("sell_max_bubble", 800))
    MIN_MATURITY = float(settings.get("min_maturity", 0.05))
    MAX_MATURITY = float(settings.get("max_maturity", 0.25))

    BUY_MIN_VOL_BUBBLE = float(settings.get("buy_min_vol_bubble", 0.0))
    BUY_MAX_VOL_BUBBLE = float(settings.get("buy_max_vol_bubble", 0.01))
    BUY_MULT_VOL_MIN = float(settings.get("buy_mult_vol_min", 2.5))
    BUY_MULT_VOL_MAX = float(settings.get("buy_mult_vol_max", 0.0))
    VOL_LOOKBACK = int(settings.get("vol_lookback", 48))

    ANGLE_UP_MIN = float(settings.get("angle_up_min", 0.1))
    ANGLE_DOWN_MIN = float(settings.get("angle_down_min", -0.5))
    ANGLE_LOOKBACK = int(settings.get("angle_lookback", 48))

    SLOPE_SALE = float(settings.get("slope_sale", 1.0))
    slope_sale = SLOPE_SALE

    BUY_MULT_TREND_UP = float(settings.get("buy_mult_trend_up", 1.0))
    BUY_MULT_TREND_FLOOR = float(settings.get("buy_mult_trend_floor", 0.25))
    BUY_MULT_TREND_DOWN = float(settings.get("buy_mult_trend_down", 0.0))

    print(
        f"[DEBUG] Loaded coin settings for {coin}: angle_up={ANGLE_UP_MIN:.2f}, "
        f"angle_down={ANGLE_DOWN_MIN:.2f}, vol_lookback={VOL_LOOKBACK}, slope_sale={SLOPE_SALE}"
    )
    default_cfg = load_coin_settings("default")
    for key, val in [
        ("angle_up_min", ANGLE_UP_MIN),
        ("angle_down_min", ANGLE_DOWN_MIN),
        ("angle_lookback", ANGLE_LOOKBACK),
        ("slope_sale", SLOPE_SALE),
    ]:
        if default_cfg.get(key) != val:
            print(
                f"[DEBUG] default {key}={default_cfg.get(key)} differs from {val} for {coin}"
            )

    # Inject settings into buy evaluation module
    buy_module.BUY_MIN_BUBBLE = BUY_MIN_BUBBLE
    buy_module.BUY_MAX_BUBBLE = BUY_MAX_BUBBLE
    buy_module.MIN_NOTE_SIZE_PCT = MIN_NOTE_SIZE_PCT
    buy_module.MAX_NOTE_SIZE_PCT = MAX_NOTE_SIZE_PCT
    buy_module.SELL_MIN_BUBBLE = SELL_MIN_BUBBLE
    buy_module.SELL_MAX_BUBBLE = SELL_MAX_BUBBLE
    buy_module.MIN_MATURITY = MIN_MATURITY
    buy_module.MAX_MATURITY = MAX_MATURITY
    buy_module.BUY_MIN_VOL_BUBBLE = BUY_MIN_VOL_BUBBLE
    buy_module.BUY_MAX_VOL_BUBBLE = BUY_MAX_VOL_BUBBLE
    buy_module.BUY_MULT_VOL_MIN = BUY_MULT_VOL_MIN
    buy_module.BUY_MULT_VOL_MAX = BUY_MULT_VOL_MAX
    buy_module.BUY_MULT_TREND_UP = BUY_MULT_TREND_UP
    buy_module.BUY_MULT_TREND_FLOOR = BUY_MULT_TREND_FLOOR
    buy_module.BUY_MULT_TREND_DOWN = BUY_MULT_TREND_DOWN

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
    capital = START_CAPITAL
    open_notes: list[dict[str, float]] = []
    last_month: tuple[int, int] | None = None

    def on_candle(idx, row):
        if not feed:
            return
        price = float(row.get("close", 0.0))
        ts_val = int(row.get("timestamp", 0))
        feed.candle(
            int(row["candle_index"]),
            ts_val,
            float(row.get("open", price)),
            float(row.get("high", price)),
            float(row.get("low", price)),
            price,
        )
        if idx >= ANGLE_LOOKBACK and idx % WINDOW_STEP == 0:
            feed.indicator(int(row["candle_index"]), "angle", float(row.get("angle", 0.0)))
        if idx >= VOL_LOOKBACK and idx % WINDOW_STEP == 0:
            feed.indicator(int(row["candle_index"]), "vol", float(row.get("volatility", 0.0)))

    def buy_handler(idx, row, notes, cap):
        return buy_module.evaluate_buy(idx, row, pts, cap, notes)

    def sell_handler(idx, row, notes, cap):
        nonlocal last_month
        ts = int(row.get("timestamp", 0))
        dt = pd.to_datetime(ts, unit="s", utc=True)
        current_month = (dt.year, dt.month)
        if current_month != last_month:
            cap += MONTHLY_TOPUP
            last_month = current_month
            log.what(
                f"Monthly top-up: +{MONTHLY_TOPUP} USDT at {dt.date()} → Capital={cap:.2f}"
            )
            action_handler("TOPUP", idx, row, account="sim", coin=coin)
        price = float(row["close"])
        angle = float(row.get("angle", 0.0))
        return evaluate_sell(idx, price, notes, cap, angle=angle, slope_sale=slope_sale)

    import json as _json
    from pathlib import Path

    def ledger_handler(trade, account, coin):
        if not trade:
            return
        if feed:
            if trade.get("side") == "BUY":
                feed.buy(
                    int(trade.get("idx", 0)),
                    float(trade.get("price", 0.0)),
                    float(trade.get("units", 0.0)),
                    float(trade.get("usd", 0.0)),
                    float(trade.get("target", 0.0)),
                )
            else:
                feed.sell(
                    int(trade.get("idx", 0)),
                    float(trade.get("price", 0.0)),
                    float(trade.get("units", 0.0)),
                    float(trade.get("usd", 0.0)),
                    float(trade.get("entry_price", 0.0)),
                )
        path = Path("data") / "ledgers" / f"{account}_{coin}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            _json.dump(trade, f)
            f.write("\n")

    def action_handler(action, idx, row, account, coin):
        path = Path("data") / "actions" / f"{account}_{coin}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"idx": int(idx), "timestamp": int(row.get("timestamp", 0)), "action": action}
        with path.open("a", encoding="utf-8") as f:
            _json.dump(record, f)
            f.write("\n")

    handlers = {
        "buy": buy_handler,
        "sell": sell_handler,
        "ledger": ledger_handler,
        "action": action_handler,
        "on_candle": on_candle,
        "capital": capital,
        "open_notes": open_notes,
    }

    capital, open_notes = run_candle_loop(df, handlers, account="sim", coin=coin)

    final_value = capital + sum(n["units"] * float(df["close"].iloc[-1]) for n in open_notes)

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
