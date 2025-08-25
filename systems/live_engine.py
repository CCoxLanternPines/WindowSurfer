from __future__ import annotations

"""Minimal live engine stub emitting graph feed."""

import json
import pathlib

import systems.scripts.evaluate_buy as buy_module
from systems.scripts.evaluate_sell import evaluate_sell
from systems.utils.graph_feed import GraphFeed
from systems.utils.config_loader import get_coin_setting


SETTINGS_PATH = pathlib.Path("settings/settings.json")

if SETTINGS_PATH.exists():
    _settings = json.loads(SETTINGS_PATH.read_text())
    START_CAPITAL = _settings.get("general_settings", {}).get("simulation_capital", 10_000)
else:
    START_CAPITAL = 10_000


def run_live(
    *,
    account: str,
    market: str,
    graph_feed: bool = False,
    graph_downsample: int = 5,
) -> None:
    coin = market.replace("/", "").upper()

    EXHAUSTION_LOOKBACK = int(get_coin_setting(coin, "exhaustion_lookback", 184))
    WINDOW_STEP = int(get_coin_setting(coin, "window_step", 12))

    BUY_MIN_BUBBLE = float(get_coin_setting(coin, "buy_min_bubble", 100))
    BUY_MAX_BUBBLE = float(get_coin_setting(coin, "buy_max_bubble", 500))
    MIN_NOTE_SIZE_PCT = float(get_coin_setting(coin, "min_note_size_pct", 0.03))
    MAX_NOTE_SIZE_PCT = float(get_coin_setting(coin, "max_note_size_pct", 0.25))

    SELL_MIN_BUBBLE = float(get_coin_setting(coin, "sell_min_bubble", 150))
    SELL_MAX_BUBBLE = float(get_coin_setting(coin, "sell_max_bubble", 800))
    MIN_MATURITY = float(get_coin_setting(coin, "min_maturity", 0.05))
    MAX_MATURITY = float(get_coin_setting(coin, "max_maturity", 0.25))

    BUY_MIN_VOL_BUBBLE = float(get_coin_setting(coin, "buy_min_vol_bubble", 0.0))
    BUY_MAX_VOL_BUBBLE = float(get_coin_setting(coin, "buy_max_vol_bubble", 0.01))
    BUY_MULT_VOL_MIN = float(get_coin_setting(coin, "buy_mult_vol_min", 2.5))
    BUY_MULT_VOL_MAX = float(get_coin_setting(coin, "buy_mult_vol_max", 0.0))
    VOL_LOOKBACK = int(get_coin_setting(coin, "vol_lookback", 48))

    ANGLE_UP_MIN = float(get_coin_setting(coin, "angle_up_min", 0.1))
    ANGLE_DOWN_MIN = float(get_coin_setting(coin, "angle_down_min", -0.5))
    ANGLE_LOOKBACK = int(get_coin_setting(coin, "angle_lookback", 48))

    SLOPE_SALE = float(get_coin_setting(coin, "slope_sale", 1.0))
    slope_sale = SLOPE_SALE

    BUY_MULT_TREND_UP = float(get_coin_setting(coin, "buy_mult_trend_up", 1.0))
    BUY_MULT_TREND_FLOOR = float(get_coin_setting(coin, "buy_mult_trend_floor", 0.25))
    BUY_MULT_TREND_DOWN = float(get_coin_setting(coin, "buy_mult_trend_down", 0.0))

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

    capital = START_CAPITAL

    # Placeholder angle computation for parity with simulation engine
    angle = 0.0
    evaluate_sell(0, 0.0, [], 0.0, angle=angle, slope_sale=slope_sale)

    feed = None
    if graph_feed:
        feed = GraphFeed(
            mode="live",
            coin=coin,
            account=account,
            downsample=graph_downsample,
            flush=True,
        )

    # Real trading logic would go here. This placeholder simply closes the feed.
    if feed:
        feed.close()
