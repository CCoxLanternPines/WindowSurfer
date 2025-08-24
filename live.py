#!/usr/bin/env python3
"""CLI entry point for live trading loop."""

from __future__ import annotations

import argparse
from pathlib import Path
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

    from systems.live_engine import run_live

    run_live(account=args.account, market=_normalize_market(args.market))

    if args.graph:
        ledger_path = Path("data/ledgers") / f"{args.account}_{args.market}.json"
        candles_path = Path("data/live") / f"{args.market}.csv"
        try:
            import graph
            graph.plot(str(ledger_path), str(candles_path))
        except Exception as exc:  # pragma: no cover - plotting best effort
            print(f"[WARN] Plotting failed: {exc}")


if __name__ == "__main__":  # pragma: no cover
    main()
