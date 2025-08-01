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
    parser.add_argument(
        "--mode",
        required=True,
        choices=["sim", "live", "wallet"],
        help="Execution mode: sim, live, or wallet",
    )
    parser.add_argument(
        "--tag",
        required=False,
        help=(
            "Symbol tag, e.g. DOGEUSD. If omitted, all symbols from config are "
            "processed every hour"
        ),
    )
    parser.add_argument(
        "--window",
        required=False,
        help="Window name (forward-compatible)",
    )
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Run live mode once immediately and exit",
    )
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
    verbose = args.verbose

    if mode == "wallet":
        from systems.scripts.kraken_utils import get_kraken_balance

        balances = get_kraken_balance(verbose)

        if verbose >= 1:
            print("[WALLET] Kraken Balance")
            if verbose >= 2:
                print(balances)
            for asset, amount in balances.items():
                val = float(amount)
                if val == 0:
                    continue
                fmt = f"{val:.2f}" if val > 1 else f"{val:.6f}"
                if asset.upper() in {"ZUSD", "USD", "USDT"}:
                    print(f"{asset}: ${fmt}")
                else:
                    print(f"{asset}: {fmt}")
        return

    if mode == "sim":
        run_simulation(tag=args.tag.upper(), verbose=args.verbose)
    elif mode == "live":
        run_live(
            tag=args.tag.upper() if args.tag else None,
            window=args.window,
            dry=args.dry,
            verbose=args.verbose,
        )
    else:
        addlog("Error: --mode must be either 'sim', 'live', or 'wallet'")
        sys.exit(1)


if __name__ == "__main__":
    main()
