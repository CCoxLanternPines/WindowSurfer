"""Command line entry point for WindowSurfer."""

from __future__ import annotations

import argparse
import sys

from systems.live_engine import run_live
from systems.sim_engine import run_simulation
from systems.scripts.loader import load_settings

SETTINGS = load_settings()


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="WindowSurfer bot entrypoint")
    parser.add_argument("--mode", required=True, help="Execution mode: sim or live")
    parser.add_argument("--tag", required=True, help="Symbol tag, e.g. DOGEUSD")
    parser.add_argument(
        "--window", required=True, help="Candle window, e.g. 1m or 1h"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity level (use -v or -vv)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run live mode in debug mode (skip countdown, run immediately)",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])

    mode = args.mode.lower()
    tag = args.tag
    window = args.window
    verbose = args.verbose

    if mode == "sim":
        run_simulation(tag=tag, window=window, verbose=verbose)
    elif mode == "live":
        run_live(tag=tag, window=window, verbose=verbose, debug=args.debug)

    else:
        print("Error: --mode must be either 'sim' or 'live'", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

