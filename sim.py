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
    parser.add_argument("--viz", action="store_true", help="Run graph_engine after simulation")
    parser.add_argument("-v", action="count", default=0, help="Increase verbosity (use -vv for more)")
    parser.add_argument("--log", action="store_true", help="Write logs to file")
    args = parser.parse_args(argv)

    coin = args.coin.replace("/", "").upper()

    init_logger(verbosity=1 + args.v, to_file=args.log, name_hint=f"sim_{coin}")
    what(f"Running simulation for {coin} with timeframe {args.time}")

    run_simulation(
        coin=coin,
        timeframe=args.time,
        viz=False,  # plotting moved to graph_engine
    )

    if args.viz:
        from systems.graph_engine import discover_feed, render_feed

        path = discover_feed(mode="sim", coin=coin)
        render_feed(path)


if __name__ == "__main__":  # pragma: no cover
    main()

