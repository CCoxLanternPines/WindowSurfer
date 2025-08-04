"""Command line entry point for WindowSurfer."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from systems.live_engine import run_live
from systems.sim_engine import run_simulation
from systems.utils.addlog import init_logger, addlog
from systems.utils.settings_loader import load_settings
from systems.utils.symbol_mapper import ensure_all_symbols_loaded


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="WindowSurfer bot entrypoint")
    parser.add_argument(
        "--mode",
        required=False,
        choices=["sim", "simtune", "live", "wallet"],
        help="Execution mode: sim, simtune, live, or wallet",
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
    parser.add_argument(
        "--clear_ledgers",
        action="store_true",
        help="Deletes all JSON files under data/ledgers after confirmation",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    init_logger(
        logging_enabled=args.log,
        verbose_level=args.verbose,
        telegram_enabled=args.telegram,
    )

    if args.clear_ledgers:
        prompt = input(
            "Are you sure you want to delete all ledger files in /data/ledgers? (y/n): "
        )
        if prompt.lower() != "y":
            print("Cancelled.")
            return
        ledger_dir = Path("data/ledgers")
        for file in ledger_dir.glob("*.json"):
            os.remove(file)
        print("✅ All ledger files deleted.")
        return

    if not args.mode:
        addlog(
            "Error: --mode must be either 'sim', 'simtune', 'live', or 'wallet'",
            verbose_int=1,
            verbose_state=args.verbose,
        )
        sys.exit(1)

    mode = args.mode.lower()
    verbose = args.verbose

    if mode == "wallet":
        from systems.scripts.kraken_utils import get_kraken_balance

        balances = get_kraken_balance(verbose)

        if verbose >= 1:
            addlog("[WALLET] Kraken Balance", verbose_int=1, verbose_state=verbose)
            addlog(str(balances), verbose_int=2, verbose_state=verbose)
            for asset, amount in balances.items():
                val = float(amount)
                if val == 0:
                    continue
                fmt = f"{val:.2f}" if val > 1 else f"{val:.6f}"
                if asset.upper() in {"ZUSD", "USD", "USDT"}:
                    addlog(f"{asset}: ${fmt}", verbose_int=1, verbose_state=verbose)
                else:
                    addlog(f"{asset}: {fmt}", verbose_int=1, verbose_state=verbose)
        return

    settings = load_settings()
    ensure_all_symbols_loaded(settings)

    if mode == "sim":
        run_simulation(tag=args.tag.upper(), verbose=args.verbose)
    elif mode == "simtune":
        if not args.window:
            addlog("Error: --window is required for simtune mode")
            sys.exit(1)
        from systems.scripts.sim_tuner import run_sim_tuner
        run_sim_tuner(tag=args.tag.upper(), window=args.window.lower(), verbose=args.verbose)
    elif mode == "live":
        run_live(
            tag=args.tag.upper() if args.tag else None,
            window=args.window,
            dry=args.dry,
            verbose=args.verbose,
        )
    else:
        addlog(
            "Error: --mode must be either 'sim', 'simtune', 'live', or 'wallet'",
            verbose_int=1,
            verbose_state=verbose,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
