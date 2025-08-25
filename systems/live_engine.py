from __future__ import annotations

"""Minimal live engine stub emitting graph feed."""

from systems.utils.graph_feed import GraphFeed


def run_live(
    *,
    account: str,
    market: str,
    graph_feed: bool = False,
    graph_downsample: int = 5,
) -> None:
    coin = market.replace("/", "").upper()
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
