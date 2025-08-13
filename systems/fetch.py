from __future__ import annotations

"""CLI to fetch historical candle data."""

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from systems.utils.addlog import addlog
from systems.utils.config import load_settings
from systems.scripts.fetch_candles import fetch_all, fetch_recent


def _resolve_symbols(coin: str) -> tuple[str, str]:
    settings = load_settings()
    for cfg in settings.get("ledger_settings", {}).values():
        kraken = cfg.get("kraken_name")
        binance = cfg.get("binance_name")
        if kraken and kraken.split("/")[0].upper() == coin.upper():
            if not binance:
                break
            return kraken, binance
    addlog(
        f"[ERROR] Coin '{coin}' not found in settings",
        verbose_int=1,
        verbose_state=True,
    )
    sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch candle data")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_all = sub.add_parser("all", help="Fetch full history from Binance")
    p_all.add_argument("coin")

    p_recent = sub.add_parser("recent", help="Fetch recent hours")
    p_recent.add_argument("coin")
    p_recent.add_argument("--hours", type=int, required=True)

    args = parser.parse_args(argv)
    coin = args.coin.upper()
    kraken_symbol, binance_symbol = _resolve_symbols(coin)

    if args.mode == "all":
        fetch_all(coin, binance_symbol)
    else:
        fetch_recent(coin, kraken_symbol, binance_symbol, args.hours)


if __name__ == "__main__":
    main()

