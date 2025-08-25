#!/usr/bin/env python3
"""CLI entry point for running historical simulations."""

from __future__ import annotations

import argparse
from typing import Optional

from systems.sim_engine import run_simulation


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run historical simulation")
    parser.add_argument("--coin", required=True, help="Coin symbol e.g. DOGEUSD")
    parser.add_argument("--time", default="1m", help="Lookback window")
    parser.add_argument("--viz", action="store_true", help="Enable plotting")
    args = parser.parse_args(argv)

    coin = args.coin.replace("/", "").upper()

    run_simulation(coin=coin, timeframe=args.time, viz=args.viz)


if __name__ == "__main__":  # pragma: no cover
    main()

