from __future__ import annotations

"""Command line entry point for WindowSurfer."""

import os
import sys

from systems.live_engine import run_live
from systems.sim_engine import run_simulation
from systems.utils.addlog import init_logger, addlog
from systems.utils.cli import build_parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    parser.add_argument(
        "--time",
        type=str,
        default=None,
        help="How far back to simulate (e.g. 1m, 7d, 1y)",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Enable visualization plotting",
    )

    args = parser.parse_args(argv or sys.argv[1:])
    if not args.mode:
        parser.error("--mode is required")
    mode = args.mode.lower()
    os.environ["WS_MODE"] = mode
    init_logger(
        logging_enabled=False,
        verbose_level=args.verbose,
        telegram_enabled=args.telegram and mode != "sim",
    )

    if mode == "sim":
        if not args.account or not args.market:
            addlog("Error: --account and --market are required for sim mode")
            sys.exit(1)
        run_simulation(
            account=args.account,
            market=args.market,
            verbose=args.verbose,
            timeframe=args.time,
            viz=args.viz,
        )
    elif mode == "live":
        if not args.account or not args.market:
            addlog("Error: --account and --market are required for live mode")
            sys.exit(1)
        run_live(
            account=args.account,
            market=args.market,
            dry=args.dry,
            verbose=args.verbose,
        )
    elif mode == "test":
        if not args.account or not args.market:
            addlog("Error: --account and --market are required for test mode")
            sys.exit(1)
        from systems.scripts.test_mode import run_test_mode

        run_test_mode(args.account, args.market)
    elif mode == "view":
        if not args.account:
            addlog("Error: --account is required for view mode")
            sys.exit(1)
        from systems.scripts.view_log import view_log

        view_log(args.account)
    else:
        parser.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
