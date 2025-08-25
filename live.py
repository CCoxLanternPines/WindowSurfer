#!/usr/bin/env python3
"""CLI entry point for live trading loop.

Example:
    python live.py --account Kris --market DOGEUSD --graph

Use ``--help`` to see available options.
"""

from __future__ import annotations

import argparse
from typing import Optional
import sys

from systems.utils.config import (
    load_account_settings,
    load_coin_settings,
    resolve_account_market,
    resolve_coin_config,
)
from systems.scripts.plot import plot_trades_from_ledger


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run live trading loop")
    parser.add_argument("--account", required=True, help="Account name")
    parser.add_argument("--market", required=True, help="Market symbol e.g. DOGEUSD")
    parser.add_argument("--graph", action="store_true", help="Plot ledger after run")
    args = parser.parse_args(argv)
    from systems.utils.load_config import load_config

    market = args.market.replace("/", "").upper()

    cfg = load_config()
    accounts = cfg.get("accounts", {})
    if args.account not in accounts:
        print(f"[ERROR] Unknown account: {args.account}")
        raise SystemExit(1)
    markets = accounts[args.account].get("markets", {})
    if market not in markets:
        print(f"[ERROR] Unknown market: {market} for account {args.account}")
        raise SystemExit(1)

    accounts_cfg = load_account_settings()
    coin_cfg = load_coin_settings()
    acct_cfg = accounts_cfg.get(args.account)
    if not acct_cfg:
        print(f"[ERROR] Unknown account {args.account}")
        sys.exit(1)
    if market not in acct_cfg.get("market settings", {}):
        print(f"[ERROR] Unknown market {market} for account {args.account}")
        sys.exit(1)
    _ = {
        **resolve_coin_config(market, coin_cfg),
        **resolve_account_market(args.account, market, accounts_cfg),
    }

    from systems.live_engine import run_live

    run_live(account=args.account, market=market)

    if args.graph:
        try:
            plot_trades_from_ledger(args.account, market, mode="live")
        except Exception as exc:  # pragma: no cover - plotting best effort
            print(f"[WARN] Plotting failed: {exc}")



if __name__ == "__main__":  # pragma: no cover
    main()
