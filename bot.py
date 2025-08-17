"""Command line entry point for WindowSurfer."""

from __future__ import annotations

import sys
from systems.utils.asset_pairs import load_asset_pairs

from systems.fetch import run_fetch
from systems.live_engine import run_live
from systems.sim_engine import run_simulation
from systems.scripts.wallet import show_wallet
from systems.utils.addlog import init_logger, addlog
from systems.utils.config import load_settings
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
    init_logger(
        logging_enabled=args.log,
        verbose_level=args.verbose,
        telegram_enabled=args.telegram,
    )

    mode = args.mode.lower()
    verbose = args.verbose

    settings = load_settings()

    try:
        asset_pairs = load_asset_pairs()
        valid_pairs = {pair_info["altname"].upper() for pair_info in asset_pairs.values()}
    except Exception:
        addlog(
            "[ERROR] Failed to load Kraken AssetPairs",
            verbose_int=1,
            verbose_state=True,
        )
        sys.exit(1)

    for ledger_cfg in settings.get("ledger_settings", {}).values():
        tag = ledger_cfg.get("tag", "")
        if tag.upper() not in valid_pairs:
            raise RuntimeError(
                f"[ERROR] Invalid trading pair: {ledger_cfg['tag']} — Not found in Kraken altname list",
            )

    if mode == "fetch":
        run_fetch(args.ledger)
    elif mode == "sim":
        if not args.ledger:
            addlog("Error: --ledger is required for sim mode")
            sys.exit(1)
        run_simulation(
            ledger=args.ledger,
            verbose=args.verbose,
            timeframe=args.time,
            viz=args.viz,
        )
    elif mode == "simdebug":
        if not args.ledger:
            addlog("Error: --ledger is required for simdebug mode")
            sys.exit(1)
        run_simulation(
            ledger=args.ledger,
            verbose=args.verbose,
            timeframe=args.time,
            dump_signals=True,
            viz=args.viz,
        )
    elif mode == "live":
        run_live(
            ledger=args.ledger,
            dry=args.dry,
            verbose=args.verbose,
        )
    elif mode == "wallet":
        show_wallet(args.ledger, verbose)
    else:
        parser.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
