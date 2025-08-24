#!/usr/bin/env python3
"""CLI entry point for live trading loop."""

from __future__ import annotations

import argparse
from typing import Optional


def _normalize_market(market: str) -> str:
    if "/" in market:
        return market
    market = market.upper()
    for quote in ("USDT", "USDC", "USD", "EUR", "GBP"):
        if market.endswith(quote):
            base = market[: -len(quote)]
            return f"{base}/{quote}"
    return market


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run live trading loop")
    parser.add_argument("--account", required=True, help="Account name")
    parser.add_argument("--market", required=True, help="Market symbol e.g. DOGEUSD")
    parser.add_argument("--graph", action="store_true", help="Plot ledger after run")
    args = parser.parse_args(argv)
    from systems.utils.load_config import load_config

    cfg = load_config()
    accounts = cfg.get("accounts", {})
    if args.account not in accounts:
        print(f"[ERROR] Unknown account: {args.account}")
        raise SystemExit(1)
    markets = accounts[args.account].get("markets", {})
    if args.market not in markets:
        print(f"[ERROR] Unknown market: {args.market} for account {args.account}")
        raise SystemExit(1)

    from systems.live_engine import run_live

    run_live(account=args.account, market=_normalize_market(args.market))

    if args.graph:
        try:
            from systems.scripts.plot import plot_trades_from_ledger
            plot_trades_from_ledger(args.account, args.market, mode="live")
        except Exception as exc:  # pragma: no cover - plotting best effort
            print(f"[WARN] Plotting failed: {exc}")


if __name__ == "__main__":  # pragma: no cover
    main()
