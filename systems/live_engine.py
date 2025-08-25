from __future__ import annotations

"""Minimal live engine stub emitting graph feed."""

import json
import pathlib

from systems.scripts.evaluate_sell import evaluate_sell
from systems.utils.graph_feed import GraphFeed
from systems.utils.settings_loader import get_coin_setting


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
    slope_sale = float(get_coin_setting(coin, "slope_sale", 1.0))
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
