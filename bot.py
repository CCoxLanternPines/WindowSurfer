"""Command line interface for WindowSurfer operations."""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

from systems.scripts.path_utils import ledger_settings_path
from typing import Optional

from systems.scripts.config_loader import load_runtime_config

_TIME_RE = re.compile(r"^\d+[dwmy]$")


def _timespan(value: str | None) -> Optional[str]:
    """Validate shorthand timespan expressions.

    Accepts values like ``1d`` for one day, ``2w`` for two weeks,
    ``3m`` for three months and ``4y`` for four years.  Returns the
    original string for passing into downstream engines.
    """

    if value is None:
        return None
    if not _TIME_RE.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "Invalid timespan format. Use numbers followed by d, w, m or y"
        )
    return value


def _setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity >= 3:
        level = logging.NOTSET
    elif verbosity == 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
    logging.basicConfig(level=level, format="%(message)s")


def _load_coin_list(ledger_name: str) -> tuple[list[str], str]:
    """Return list of coin symbols and fiat currency for ``ledger_name``."""

    path = ledger_settings_path(ledger_name)
    if not path.exists():
        raise FileNotFoundError(f"ledger settings not found: {path}")
    import json

    with open(path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    fiat = cfg.get("fiat", "USD")
    coins = list(cfg.get("coins", {}).keys())
    return coins, fiat


def run_fetch(
    ledger_name: Optional[str],
    *,
    full: bool = False,
    update: bool = False,
    wallet_cache: bool = False,
) -> None:
    """Run fetch operations based on provided flags."""

    from systems.scripts.fetch_core import (
        build_wallet_cache,
        fetch_full_history,
        fetch_update_history,
    )

    if wallet_cache:
        build_wallet_cache()

    if not (full or update):
        return

    if not ledger_name:
        raise ValueError("--ledger is required when using --full or --update")

    coins, fiat = _load_coin_list(ledger_name)
    for symbol in coins:
        if full:
            fetch_full_history(symbol, fiat)
        if update:
            fetch_update_history(symbol, fiat)


def main() -> None:
    parser = argparse.ArgumentParser(description="WindowSurfer command line interface")
    parser.add_argument("--mode", required=True, choices=["sim", "live", "fetch"], help="Operation mode")
    parser.add_argument("--ledger", help="Ledger name (without .json)")
    parser.add_argument("--start", type=_timespan, help="Start offset from latest candle (e.g. 1d, 2w)")
    parser.add_argument("--range", type=_timespan, help="Window size for simulation (e.g. 3m)")
    parser.add_argument("--dry-run", action="store_true", help="Live mode: don't execute trades")
    parser.add_argument("--full", action="store_true", help="Fetch full Binance history")
    parser.add_argument("--update", action="store_true", help="Fetch recent Kraken candles")
    parser.add_argument(
        "--wallet_cache",
        action="store_true",
        help="Fetch mode: refresh exchange pair metadata",
    )
    parser.add_argument("-v", action="count", default=0, help="Increase verbosity (-v, -vv, -vvv)")

    args = parser.parse_args()

    _setup_logging(args.v)

    if args.mode in {"sim", "live"}:
        if not args.ledger:
            parser.error("--ledger is required for sim and live modes")
        try:
            load_runtime_config(args.ledger)
        except FileNotFoundError:
            print("Wallet cache missing. Run 'python bot.py --mode fetch --wallet_cache' first.")
            sys.exit(1)

    if args.mode == "sim":
        from systems.sim_engine import run as run_sim

        run_sim(args.ledger, start=args.start, range=args.range)
    elif args.mode == "live":
        from systems.live_engine import run as run_live

        run_live(args.ledger, dry_run=args.dry_run)
    elif args.mode == "fetch":
        try:
            run_fetch(
                args.ledger,
                full=args.full,
                update=args.update,
                wallet_cache=args.wallet_cache,
            )
        except ValueError as exc:
            parser.error(str(exc))
        except FileNotFoundError as exc:
            parser.error(str(exc))
    else:
        print("Unknown mode. Use sim, live, or fetch.")


if __name__ == "__main__":
    main()
