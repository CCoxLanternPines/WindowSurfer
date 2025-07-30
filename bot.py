"""Command line entry point for WindowSurfer."""

from __future__ import annotations

import argparse
import sys

from systems.live_engine import run_live
from systems.sim_engine import run_simulation, run_top_hour_loop
from systems.utils.logger import init_logger, addlog


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="WindowSurfer bot entrypoint")
    parser.add_argument("--mode", required=True, help="Execution mode: sim, live or top")
    parser.add_argument("--tag", required=False, help="Symbol tag, e.g. DOGEUSD")
    parser.add_argument("--window", required=False, default="3d", help="Candle window, e.g. 1h or 3d")
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity level (use -v or -vv)",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Enable log file output",
    )
    parser.add_argument(
        "--telegram",
        action="store_true",
        help="Enable Telegram alerts",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    init_logger(
        logging_enabled=args.log,
        verbose_level=args.verbose,
        telegram_enabled=args.telegram,
    )

    mode = args.mode.lower()
    tag = args.tag.upper() if args.tag else None
    window = args.window
    verbose = args.verbose

    if mode == "sim":
        if not tag:
            addlog("Error: --tag is required for simulation")
            sys.exit(1)
        run_simulation(tag=tag, window=window, verbose=verbose)
    elif mode == "live":
        if not tag:
            addlog("Error: --tag is required for live mode")
            sys.exit(1)
        run_live(tag=tag, window=window, verbose=verbose)
    elif mode == "top":
        run_top_hour_loop(tag=tag, window=window, verbose=verbose)
    else:
        addlog("Error: --mode must be 'sim', 'live', or 'top'")
        sys.exit(1)


if __name__ == "__main__":
    main()
