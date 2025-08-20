"""Command line entry point for WindowSurfer."""

from __future__ import annotations

import os
import sys

from systems.fetch import run_fetch
from systems.live_engine import run_live
from systems.sim_engine import run_simulation
from systems.scripts.wallet import show_wallet
from systems.utils.addlog import init_logger, addlog
from systems.utils.load_config import load_config
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
    parser.add_argument(
        "--period",
        type=str,
        default="daily",
        help="Report period for email mode (daily|weekly|monthly|yearly|test)",
    )

    args = parser.parse_args(argv or sys.argv[1:])
    if not args.mode:
        parser.error("--mode is required")
    mode = args.mode.lower()
    os.environ["WS_MODE"] = mode
    init_logger(
        logging_enabled=args.log,
        verbose_level=args.verbose,
    )

    verbose = args.verbose
    cfg = load_config()
    run_all = args.all or not args.account

    if mode == "fetch":
        run_fetch(account=args.account, market=args.market, all_accounts=run_all)

    elif mode == "sim":
        run_simulation(
            account=args.account,
            market=args.market,
            all_accounts=run_all,
            verbose=args.verbose,
            timeframe=args.time,
            viz=args.viz,
        )

    elif mode == "live":
        run_live(
            account=args.account,
            market=args.market,
            all_accounts=run_all,
            dry=args.dry,
            verbose=args.verbose,
        )

    elif mode == "test":
        from systems.scripts.test_mode import run_test

        if not args.ledger:
            addlog("Error: --ledger is required for test mode")
            sys.exit(1)
        exit_code = run_test(args.ledger)
        sys.exit(exit_code)

    elif mode == "wallet":
        accounts_cfg = cfg.get("accounts", {})
        targets = accounts_cfg.keys() if run_all else [args.account]
        for acct in targets:
            acct_cfg = accounts_cfg.get(acct)
            if not acct_cfg:
                addlog(
                    f"[ERROR] Unknown account {acct}",
                    verbose_int=1,
                    verbose_state=True,
                )
                continue
            os.environ["WS_ACCOUNT"] = acct
            markets_cfg = acct_cfg.get("markets", {})
            m_targets = [args.market] if args.market else markets_cfg.keys()
            for m in m_targets:
                if m not in markets_cfg:
                    continue
                show_wallet(acct, m, verbose)

    elif mode == "view":
        if not args.account:
            addlog("Error: --account is required for view mode")
            sys.exit(1)
        from systems.scripts.view_log import view_log
        view_log(args.account, timeframe=args.time)

    elif mode == "email":
        from systems.scripts.report import run_report

        if not args.account:
            addlog("Error: --account is required for email mode")
            sys.exit(1)
        run_report(args.account, args.period)

    else:
        parser.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
