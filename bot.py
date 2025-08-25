#!/usr/bin/env python3
"""Unified CLI entry point for WindowSurfer.

This optional wrapper exposes subcommands for the individual scripts so
they can be invoked from one location::

    python bot.py sim --coin DOGEUSD --time 1m --viz
    python bot.py live --account Kris --market DOGEUSD --graph
    python bot.py test --account Kris --market DOGEUSD --smoketest

Each subcommand forwards its arguments to the corresponding module.
Use ``--help`` on the subcommands to see available options.
"""

from __future__ import annotations

import argparse
from typing import Optional

import live
import sim
import test as test_mod


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level argument parser with subcommands."""

    parser = argparse.ArgumentParser(description="WindowSurfer CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sim_parser = sub.add_parser("sim", help="Run historical simulation")
    sim_parser.add_argument("--coin", required=True, help="Coin symbol e.g. DOGEUSD")
    sim_parser.add_argument("--time", default="1m", help="Lookback window")
    sim_parser.add_argument("--viz", action="store_true", help="Enable plotting")

    live_parser = sub.add_parser("live", help="Run live trading loop")
    live_parser.add_argument("--account", required=True, help="Account name")
    live_parser.add_argument("--market", required=True, help="Market symbol e.g. DOGEUSD")
    live_parser.add_argument("--graph", action="store_true", help="Plot ledger after run")

    test_parser = sub.add_parser("test", help="Run configuration tests")
    test_parser.add_argument("--account", required=True, help="Account name")
    test_parser.add_argument("--market", required=True, help="Market symbol e.g. DOGEUSD")
    test_parser.add_argument("--smoketest", action="store_true", help="Run exchange ping")

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "sim":
        sim_args = ["--coin", args.coin, "--time", args.time]
        if args.viz:
            sim_args.append("--viz")
        sim.main(sim_args)
    elif args.command == "live":
        live_args = ["--account", args.account, "--market", args.market]
        if args.graph:
            live_args.append("--graph")
        live.main(live_args)
    elif args.command == "test":
        test_args = ["--account", args.account, "--market", args.market]
        if args.smoketest:
            test_args.append("--smoketest")
        test_mod.main(test_args)
    else:  # pragma: no cover - argparse ensures we don't reach here
        parser.error(f"Unknown command {args.command}")


if __name__ == "__main__":  # pragma: no cover
    main()

