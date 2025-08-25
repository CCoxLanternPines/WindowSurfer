from __future__ import annotations

"""Live trading loop sharing candle logic with simulation."""

import json
import pathlib
import time
import logging

import ccxt  # type: ignore
import pandas as pd
import numpy as np

import systems.scripts.evaluate_buy as buy_module
from systems.scripts.evaluate_sell import evaluate_sell
from systems.scripts.candle_loop import run_candle_loop
from systems.utils.config_loader import load_coin_settings
from systems.scripts.ledger import write_trade
from systems.scripts.action_writer import write_action
from systems.utils.graph_feed import GraphFeed


SETTINGS_PATH = pathlib.Path("settings/settings.json")

if SETTINGS_PATH.exists():
    _settings = json.loads(SETTINGS_PATH.read_text())
    _general = _settings.get("general_settings", {})
    START_CAPITAL = _general.get("simulation_capital", 10_000)
    MONTHLY_TOPUP = _general.get("monthly_topup", 0)
else:
    START_CAPITAL = 10_000
    MONTHLY_TOPUP = 0


logger = logging.getLogger(__name__)
def run_live(*, account: str, market: str, place_orders: bool = False) -> None:
    """Run the live trading loop.

    Parameters
    ----------
    account:
        Account name.
    market:
        Market symbol e.g. ``DOGEUSD``.
    place_orders:
        When ``True`` orders are actually sent to Kraken.
    """
    coin = market.replace("/", "").upper()
    feed = GraphFeed(mode="live", coin=coin, account=account, flush=True)

    # ------------------------------------------------------------ dry run
    if not place_orders:
        path = pathlib.Path("data") / "sim" / f"{coin}.csv"
        if path.exists():
            df = pd.read_csv(path).tail(200)
        else:
            now = int(time.time())
            ts = [now + i * 60 for i in range(200)]
            price = 1.0
            df = pd.DataFrame(
                {
                    "timestamp": ts,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                }
            )
        df["candle_index"] = range(len(df))

        capital = START_CAPITAL
        open_notes: list[dict[str, float]] = []
        usd = 5.0
        entry_price = 0.0
        units = 0.0
        buy_idx: int | None = None

        for i, row in df.iterrows():
            ts_val = int(row.get("timestamp", 0))
            price = float(row.get("close", 0.0))
            feed.candle(
                int(row["candle_index"]),
                ts_val,
                float(row.get("open", price)),
                float(row.get("high", price)),
                float(row.get("low", price)),
                price,
            )

            sells = buys = 0
            if buy_idx is None:
                entry_price = price
                units = usd / price
                target = price * 1.01
                feed.buy(int(i), price, units, usd, target)
                write_trade(
                    {
                        "idx": int(i),
                        "price": price,
                        "side": "BUY",
                        "usd": usd,
                        "units": units,
                        "target": target,
                    },
                    account,
                    coin,
                )
                open_notes.append(
                    {"entry_price": entry_price, "units": units, "sell_price": target}
                )
                capital -= usd
                buys = 1
                buy_idx = int(i)
            elif buy_idx is not None and i >= buy_idx + 3 and open_notes:
                proceeds = usd * (price / entry_price)
                feed.sell(int(i), price, units, proceeds, entry_price)
                write_trade(
                    {
                        "idx": int(i),
                        "price": price,
                        "side": "SELL",
                        "usd": proceeds,
                        "units": units,
                        "entry_price": entry_price,
                    },
                    account,
                    coin,
                )
                open_notes.clear()
                capital += proceeds
                sells = 1

            write_action(i, row, sells, buys, open_notes, capital, account, coin)
            time.sleep(0.1)

        feed.close()
        return

    # ------------------------------------------------------------ real mode
    settings = load_coin_settings(coin)
    EXHAUSTION_LOOKBACK = int(settings.get("exhaustion_lookback", 184))
    WINDOW_STEP = int(settings.get("window_step", 12))
    VOL_LOOKBACK = int(settings.get("vol_lookback", 48))
    ANGLE_LOOKBACK = int(settings.get("angle_lookback", 48))
    SLOPE_SALE = float(settings.get("slope_sale", 1.0))
    slope_sale = SLOPE_SALE

    buy_module.BUY_MIN_BUBBLE = float(settings.get("buy_min_bubble", 100))
    buy_module.BUY_MAX_BUBBLE = float(settings.get("buy_max_bubble", 500))
    buy_module.MIN_NOTE_SIZE_PCT = float(settings.get("min_note_size_pct", 0.03))
    buy_module.MAX_NOTE_SIZE_PCT = float(settings.get("max_note_size_pct", 0.25))
    buy_module.SELL_MIN_BUBBLE = float(settings.get("sell_min_bubble", 150))
    buy_module.SELL_MAX_BUBBLE = float(settings.get("sell_max_bubble", 800))
    buy_module.MIN_MATURITY = float(settings.get("min_maturity", 0.05))
    buy_module.MAX_MATURITY = float(settings.get("max_maturity", 0.25))
    buy_module.BUY_MIN_VOL_BUBBLE = float(settings.get("buy_min_vol_bubble", 0.0))
    buy_module.BUY_MAX_VOL_BUBBLE = float(settings.get("buy_max_vol_bubble", 0.01))
    buy_module.BUY_MULT_VOL_MIN = float(settings.get("buy_mult_vol_min", 2.5))
    buy_module.BUY_MULT_VOL_MAX = float(settings.get("buy_mult_vol_max", 0.0))
    buy_module.BUY_MULT_TREND_UP = float(settings.get("buy_mult_trend_up", 1.0))
    buy_module.BUY_MULT_TREND_FLOOR = float(settings.get("buy_mult_trend_floor", 0.25))
    buy_module.BUY_MULT_TREND_DOWN = float(settings.get("buy_mult_trend_down", 0.0))

    print(
        f"[DEBUG] Loaded coin settings for {coin}: angle_lookback={ANGLE_LOOKBACK}, "
        f"vol_lookback={VOL_LOOKBACK}, slope_sale={SLOPE_SALE}"
    )

    exchange = ccxt.kraken({"enableRateLimit": True})
    capital = START_CAPITAL
    open_notes: list[dict[str, float]] = []
    last_ts = 0
    last_month: tuple[int, int] | None = None

    def on_candle(idx: int, row: pd.Series) -> None:
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
            feed.indicator(
                int(row["candle_index"]), "vol", float(row.get("volatility", 0.0))
            )

    def ledger_handler(trade, account, coin):
        write_trade(trade, account, coin)
        if not trade:
            return
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

    while True:
        ohlcv = exchange.fetch_ohlcv(market, timeframe="1h", limit=720) or []
        if not ohlcv:
            time.sleep(60)
            continue
        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = (
            pd.to_datetime(df["timestamp"], unit="ms", utc=True).astype("int64")
            // 1_000_000_000
        )
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["candle_index"] = range(len(df))
        df["returns"] = df["close"].pct_change()
        df["volatility"] = df["returns"].rolling(VOL_LOOKBACK).std().fillna(0)
        df["angle"] = 0.0

        pts = {
            "exhaustion_up": {"x": [], "y": [], "s": []},
            "exhaustion_down": {"x": [], "y": [], "s": []},
        }
        for t in range(EXHAUSTION_LOOKBACK, len(df), WINDOW_STEP):
            now_price = float(df["close"].iloc[t])
            past_price = float(df["close"].iloc[t - EXHAUSTION_LOOKBACK])
            end_idx = int(df["candle_index"].iloc[t])
            if now_price < past_price:
                delta_down = past_price - now_price
                norm_down = delta_down / max(1e-9, past_price)
                size = 1_000_000 * (norm_down ** 3)
                pts["exhaustion_down"]["x"].append(end_idx)
                pts["exhaustion_down"]["y"].append(now_price)
                pts["exhaustion_down"]["s"].append(size)

        for t in range(ANGLE_LOOKBACK, len(df)):
            dy = df["close"].iloc[t] - df["close"].iloc[t - ANGLE_LOOKBACK]
            dx = ANGLE_LOOKBACK
            angle = np.arctan2(dy, dx)
            df.at[t, "angle"] = max(-1.0, min(1.0, angle / (np.pi / 4)))

        subset = df[df["timestamp"] > last_ts]
        if subset.empty:
            now = int(time.time())
            time.sleep(max(0, 3600 - (now % 3600)))
            continue

        for idx, row in subset.iterrows():
            ts_val = int(row.get("timestamp", 0))
            dt = pd.to_datetime(ts_val, unit="s", utc=True)
            current_month = (dt.year, dt.month)
            if current_month != last_month:
                if MONTHLY_TOPUP:
                    capital += MONTHLY_TOPUP
                    print(
                        f"[INFO] Monthly top-up: +{MONTHLY_TOPUP} USDT at {dt.date()} → Capital={capital:.2f}"
                    )
                write_action(idx, row, 0, 0, open_notes, capital, account, coin)
                last_month = current_month

            sells = 0
            buys = 0

            def buy_handler(i, r, notes, cap):
                nonlocal buys
                trade, cap, notes = buy_module.evaluate_buy(i, r, pts, cap, notes)
                if trade:
                    amount = trade.get("usd", 0.0) / max(1e-9, trade.get("price", 0.0))
                    try:
                        exchange.create_order(market, "market", "buy", amount)
                    except Exception as exc:  # pragma: no cover - network
                        logger.warning("Kraken buy failed: %s", exc)
                    buys += 1
                return trade, cap, notes

            def sell_handler(i, r, notes, cap):
                nonlocal sells
                price = float(r["close"])
                angle = float(r.get("angle", 0.0))
                closed, cap, notes = evaluate_sell(
                    i, price, notes, cap, angle=angle, slope_sale=slope_sale
                )
                for t in closed:
                    amount = t.get("usd", 0.0) / max(1e-9, t.get("price", 0.0))
                    try:
                        exchange.create_order(market, "market", "sell", amount)
                    except Exception as exc:  # pragma: no cover - network
                        logger.warning("Kraken sell failed: %s", exc)
                sells += len(closed)
                return closed, cap, notes

            handlers = {
                "buy": buy_handler,
                "sell": sell_handler,
                "ledger": ledger_handler,
                "action": lambda *args, **kwargs: None,
                "on_candle": on_candle,
                "capital": capital,
                "open_notes": open_notes,
            }

            candle_df = subset.loc[[idx]]
            capital, open_notes = run_candle_loop(candle_df, handlers, account, coin)
            write_action(idx, row, sells, buys, open_notes, capital, account, coin)
            print(
                f"[DEBUG] {account}/{coin} @ {ts_val}: sells={sells}, buys={buys}, "
                f"open_notes={len(open_notes)}, capital={capital:.2f}"
            )

        last_ts = int(subset["timestamp"].iloc[-1])
        now = int(time.time())
        time.sleep(max(0, 3600 - (now % 3600)))

    feed.close()
