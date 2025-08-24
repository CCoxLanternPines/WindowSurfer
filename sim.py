#!/usr/bin/env python3
"""CLI entry point for running historical simulations."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional


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
    # Default location as per new layout
    candles_dir = Path("data/candles/sim")
    csv_path = candles_dir / f"{market}.csv"

    # Backwards compatibility with previous layout
    if not csv_path.exists():
        candles_dir = Path("data/sim")
        csv_path = candles_dir / f"{market}.csv"

    if csv_path.exists():
        return csv_path

    from systems.scripts.fetch_candles import fetch_binance_full_history_1h

    # Best effort symbol normalisation for Binance
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

    _ensure_candles(args.account, args.market)

    from systems.sim_engine import run_simulation

    run_simulation(
        account=args.account,
        market=_normalize_market(args.market),
        timeframe=args.timeframe,
        viz=args.viz,
    )

    ledger_name = f"{args.account}_{args.market}"
    source = Path("data/ledgers") / f"{ledger_name}.json"
    dest = Path("data/ledgers") / "ledger_simulation.json"
    if source.exists():
        dest.write_text(source.read_text())

    if args.viz:
        try:
            from systems.scripts.plot import plot_trades_from_ledger
            plot_trades_from_ledger(args.account, args.market, mode="sim")
        except Exception as exc:  # pragma: no cover - plotting best effort
            print(f"[WARN] Plotting failed: {exc}")


if __name__ == "__main__":  # pragma: no cover
    main()
