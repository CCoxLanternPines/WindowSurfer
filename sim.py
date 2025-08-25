#!/usr/bin/env python3
"""CLI entry point for running historical simulations."""

from __future__ import annotations

import argparse
from typing import Optional

from systems.sim_engine import run_simulation
from systems.utils.log import init_logger, what


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run historical simulation")
    parser.add_argument("--coin", required=True, help="Coin symbol e.g. DOGEUSD")
    parser.add_argument("--time", default="1m", help="Lookback window")
    parser.add_argument("--graph-feed", action="store_true", default=True, help="Emit NDJSON graph feed for graph_engine")
    parser.add_argument("--graph-downsample", type=int, default=1, help="Downsample factor for feed")
    parser.add_argument("-v", action="count", default=0, help="Increase verbosity (use -vv for more)")
    parser.add_argument("--log", action="store_true", help="Write logs to file")
    args = parser.parse_args(argv)

    coin = args.coin.replace("/", "").upper()

    init_logger(verbosity=1 + args.v, to_file=args.log, name_hint=f"sim_{coin}")
    what(f"Running simulation for {coin} with timeframe {args.time}")

    run_simulation(
        coin=coin,
        timeframe=args.time,
        graph_feed=args.graph_feed,
        graph_downsample=args.graph_downsample,
        viz=False,  # plotting moved to graph_engine
    )


if __name__ == "__main__":  # pragma: no cover
    main()

