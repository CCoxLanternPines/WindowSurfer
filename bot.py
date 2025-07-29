"""Command line entry point for WindowSurfer."""

from __future__ import annotations

import argparse
import sys

from systems.live_engine import run_live
from systems.sim_engine import run_simulation
from systems.utils.logger import init_logger, addlog


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="WindowSurfer bot entrypoint")
    parser.add_argument("--mode", required=True, help="Execution mode: sim or live")
    parser.add_argument("--tag", required=True, help="Symbol tag, e.g. DOGEUSD")
    parser.add_argument("--window", required=True, help="Candle window, e.g. 1m or 1h")
    class VerbosityAction(argparse.Action):
        """Allow ``-v`` flags to stack and ``--verbose`` to accept a level."""

        def __call__(self, parser, namespace, values, option_string=None):
            level = getattr(namespace, self.dest, 0)
            if values is None:
                setattr(namespace, self.dest, level + 1)
            else:
                try:
                    setattr(namespace, self.dest, int(values))
                except ValueError:
                    parser.error("--verbose level must be an integer")

    parser.add_argument(
        "-v",
        "--verbose",
        nargs="?",
        action=VerbosityAction,
        default=0,
        help="Verbosity level (use -v/-vv or --verbose N)",
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
    tag = args.tag
    window = args.window
    verbose = args.verbose

    if mode == "sim":
        run_simulation(tag=tag, window=window, verbose=verbose)
    elif mode == "live":
        run_live(tag=tag, window=window, verbose=verbose)

    else:
        addlog("Error: --mode must be either 'sim' or 'live'")
        sys.exit(1)


if __name__ == "__main__":
    main()
