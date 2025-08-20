"""Command line entry point for WindowSurfer."""

from __future__ import annotations

import os
import sys
from systems.utils.asset_pairs import load_asset_pairs

from systems.fetch import run_fetch
from systems.live_engine import run_live
from systems.sim_engine import run_simulation
from systems.scripts.wallet import show_wallet
from systems.utils.addlog import init_logger, addlog
from systems.utils.load_config import load_config
from systems.utils.cli import build_parser
from systems.utils.resolve_symbol import resolve_symbols, to_tag


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
        logging_enabled=args.log,
        verbose_level=args.verbose,
        telegram_enabled=args.telegram and mode != "sim",
    )

    verbose = args.verbose

    if args.ledger and not args.account:
        addlog(
            "[DEPRECATED] --ledger is deprecated; use --account",
            verbose_int=1,
            verbose_state=True,
        )
        args.account = args.ledger

    cfg = load_config()

    try:
        asset_pairs = load_asset_pairs()
        valid_pairs = {
            to_tag(pair_info.get("wsname", "")) for pair_info in asset_pairs.values()
        }
    except Exception:
        addlog(
            "[ERROR] Failed to load Kraken AssetPairs",
            verbose_int=1,
            verbose_state=True,
        )
        sys.exit(1)

    account_cfg = None
    check_markets: list[str] = []
    if args.account:
        account_cfg = cfg.get("accounts", {}).get(args.account)
        if not account_cfg:
            addlog(
                f"[ERROR] Unknown account {args.account}",
                verbose_int=1,
                verbose_state=True,
            )
            sys.exit(1)
        markets = account_cfg.get("markets", {})
        if args.market and args.market not in markets:
            addlog(
                f"[ERROR] Market {args.market} not configured for account {args.account}",
                verbose_int=1,
                verbose_state=True,
            )
            sys.exit(1)
        check_markets = [args.market] if args.market else list(markets.keys())
        for m in check_markets:
            symbols = resolve_symbols(m)
            tag = to_tag(symbols["kraken_name"])
            if tag.upper() not in valid_pairs:
                addlog(
                    f"[ERROR] Invalid trading pair: {m} — Not found in Kraken altname list",
                    verbose_int=1,
                    verbose_state=True,
                )

    if mode == "fetch":
        if not args.account:
            addlog("Error: --account is required for fetch mode")
            sys.exit(1)
        run_fetch(args.account, market=args.market)
    elif mode == "sim":
        if not args.account:
            addlog("Error: --account is required for sim mode")
            sys.exit(1)
        markets_to_run = check_markets if args.account else []
        for m in markets_to_run:
            run_simulation(
                account=args.account,
                market=m,
                verbose=args.verbose,
                timeframe=args.time,
                viz=args.viz,
            )
    elif mode == "live":
        if not args.account:
            addlog("Error: --account is required for live mode")
            sys.exit(1)
        run_live(
            account=args.account,
            market=args.market,
            dry=args.dry,
            verbose=args.verbose,
        )
    elif mode == "wallet":
        if not args.account:
            addlog("Error: --account is required for wallet mode")
            sys.exit(1)
        show_wallet(args.account, args.market, verbose)
    else:
        parser.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
