#!/usr/bin/env python3
"""CLI entry point for running historical simulations."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional
import json

from systems.scripts.plot import plot_from_json


def _ensure_candles(coin: str) -> Path:
    """Ensure candle CSV exists for ``coin``.

    If the canonical file is missing, attempt to fetch it from Binance.
    """
    coin = coin.replace("/", "").upper()
    candles_dir = Path("data/candles/sim")
    csv_path = candles_dir / f"{coin}.csv"

    # Backwards compatibility with previous layout
    if not csv_path.exists():
        candles_dir = Path("data/sim")
        csv_path = candles_dir / f"{coin}.csv"

    if csv_path.exists():
        return csv_path

    from systems.scripts.fetch_candles import fetch_binance_full_history_1h

    symbol = coin
    if symbol.endswith("USD"):
        symbol = symbol + "T"
    df = fetch_binance_full_history_1h(symbol)
    candles_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return csv_path


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run historical simulation")
    parser.add_argument("--coin", required=True, help="Coin symbol e.g. DOGEUSD")
    parser.add_argument("--time", default="1m", help="Lookback window")
    parser.add_argument("--viz", action="store_true", help="Enable plotting")
    args = parser.parse_args(argv)

    coin = args.coin.replace("/", "").upper()

    _ensure_candles(coin)

    from systems.sim_engine import run_simulation

    run_simulation(
        coin=coin,
        timeframe=args.time,
        viz=False,
    )
    sim_path = Path("data/temp/sim_data.json")
    print(f"[DEBUG][sim.py] Looking for sim ledger at {sim_path.resolve()}")
    if not sim_path.exists():
        print(f"[ERROR] Simulation did not produce {sim_path}")
        return

    with sim_path.open("r", encoding="utf-8") as fh:
        ledger = json.load(fh)

    final_val = ledger.get("meta", {}).get("final_value")
    if final_val is not None:
        print(f"Final Value (USD): ${float(final_val):.2f}")

    if args.viz:
        try:
            plot_from_json(str(sim_path))
        except Exception as exc:  # pragma: no cover - plotting best effort
            print(f"[WARN] Plotting failed: {exc}")


if __name__ == "__main__":  # pragma: no cover
    main()
