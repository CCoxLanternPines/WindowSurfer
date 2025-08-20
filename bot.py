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

    cfg = load_config()

    if getattr(args, "ledger", None):
        addlog(
            "[DEPRECATED] --ledger is deprecated; use --account",
            verbose_int=1,
            verbose_state=True,
        )
        if not args.account:
            args.account = args.ledger

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

    accounts_cfg = cfg.get("accounts", {})
    if args.all or not args.account:
        account_list = list(accounts_cfg.keys())
    else:
        if args.account not in accounts_cfg:
            addlog(
                f"[ERROR] Unknown account {args.account}",
                verbose_int=1,
                verbose_state=True,
            )
            sys.exit(1)
        account_list = [args.account]

    run_map: dict[str, list[str]] = {}
    for acct in account_list:
        markets_cfg = accounts_cfg.get(acct, {}).get("markets", {})
        if args.market:
            if args.market not in markets_cfg:
                if args.account:
                    addlog(
                        f"[ERROR] Market {args.market} not configured for account {acct}",
                        verbose_int=1,
                        verbose_state=True,
                    )
                    sys.exit(1)
                continue
            markets = [args.market]
        else:
            markets = list(markets_cfg.keys())
        for m in markets:
            symbols = resolve_symbols(m)
            tag = to_tag(symbols["kraken_name"])
            if tag.upper() not in valid_pairs:
                addlog(
                    f"[ERROR] Invalid trading pair: {m} — Not found in Kraken altname list",
                    verbose_int=1,
                    verbose_state=True,
                )
                sys.exit(1)
        if markets:
            run_map[acct] = markets

    if mode == "fetch":
        for acct, markets in run_map.items():
            for m in markets:
                run_fetch(acct, market=m)
    elif mode == "sim":
        for acct, markets in run_map.items():
            for m in markets:
                run_simulation(
                    account=acct,
                    market=m,
                    verbose=args.verbose,
                    timeframe=args.time,
                    viz=args.viz,
                )
    elif mode == "live":
        run_live(
            account=None if (args.all or not args.account) else args.account,
            market=args.market,
            dry=args.dry,
            verbose=args.verbose,
        )
    elif mode == "wallet":
        for acct, markets in run_map.items():
            for m in markets:
                os.environ["WS_ACCOUNT"] = acct
                show_wallet(acct, m, verbose)
    else:
        parser.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
