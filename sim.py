#!/usr/bin/env python3
"""CLI entry point for running historical simulations."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional
import sys

from systems.utils.config import (
    load_account_settings,
    load_coin_settings,
    resolve_account_market,
    resolve_coin_config,
)
from systems.scripts.plot import plot_trades_from_ledger


def _normalize_market(market: str) -> str:
    """Convert markets like ``DOGEUSD`` to ``DOGE/USD``."""
    if "/" in market:
        return market
    market = market.upper()
    for quote in ("USDT", "USDC", "USD", "EUR", "GBP"):
        if market.endswith(quote):
            base = market[: -len(quote)]
            return f"{base}/{quote}"
    return market


def _ensure_candles(account: str, market: str) -> Path:
    """Ensure candle CSV exists for ``market``.

    If the canonical file is missing, attempt to fetch it using the
    ``systems.scripts.fetch_candles`` helpers.
    """
    candles_dir = Path("data/candles/sim")
    csv_path = candles_dir / f"{market}.csv"

    # Backwards compatibility with previous layout
    if not csv_path.exists():
        candles_dir = Path("data/sim")
        csv_path = candles_dir / f"{market}.csv"

    if csv_path.exists():
        return csv_path

    from systems.scripts.fetch_candles import fetch_binance_full_history_1h

    symbol = market.upper()
    if symbol.endswith("USD"):
        symbol = symbol + "T"
    df = fetch_binance_full_history_1h(symbol)
    candles_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return csv_path


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run historical simulation")
    parser.add_argument("--account", required=True, help="Account name")
    parser.add_argument("--market", required=True, help="Market symbol e.g. DOGEUSD")
    parser.add_argument("--time", dest="timeframe", default="1m", help="Lookback window")
    parser.add_argument("--viz", action="store_true", help="Enable plotting")
    args = parser.parse_args(argv)

    # Load and validate settings
    accounts_cfg = load_account_settings()
    coin_cfg = load_coin_settings()

    acct_cfg = accounts_cfg.get(args.account)
    if not acct_cfg:
        print(f"[ERROR] Unknown account {args.account}")
        sys.exit(1)
    if args.market not in acct_cfg.get("market settings", {}):
        print(f"[ERROR] Unknown market {args.market} for account {args.account}")
        sys.exit(1)

    # Merge configs to ensure consistency (unused here but placeholder for future)
    _ = {
        **resolve_coin_config(args.market, coin_cfg),
        **resolve_account_market(args.account, args.market, accounts_cfg),
    }

    csv_path = _ensure_candles(args.account, args.market)

    from systems.sim_engine import run_simulation

    normalized_market = _normalize_market(args.market)
    run_simulation(
        account=args.account,
        market=normalized_market,
        timeframe=args.timeframe,
        viz=False,
    )
    sim_path = Path("data/temp/sim_data.json")
    print(f"[DEBUG][sim.py] Looking for sim ledger at {sim_path.resolve()}")
    if not sim_path.exists():
        print(f"[ERROR] Simulation did not produce {sim_path}")
        return

    if args.viz:
        try:
            plot_trades_from_ledger(
                args.account,
                args.market,
                mode="sim",
                ledger_path=str(sim_path),
            )
        except Exception as exc:  # pragma: no cover - plotting best effort
            print(f"[WARN] Plotting failed: {exc}")


if __name__ == "__main__":  # pragma: no cover
    main()
