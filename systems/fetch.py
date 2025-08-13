from __future__ import annotations

"""Utilities for fetching and storing market data."""

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from systems.utils.addlog import addlog
from systems.utils.config import resolve_ccxt_symbols_by_coin
from systems.scripts.fetch_candles import load_coin_csv
from systems.scripts.fetch_core import (
    COLUMNS,
    fetch_binance_range,
    fetch_kraken_range,
    get_coin_raw_path,
)


def _save_df(coin: str, df: pd.DataFrame) -> int:
    """Save ``df`` to the canonical raw path for ``coin``."""

    path = get_coin_raw_path(coin)
    tmp = path.with_suffix(".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tmp, index=False)
    tmp.replace(path)
    return len(df)


def fetch_all(coin: str) -> int:
    """Fetch full Binance history for ``coin`` and write to disk."""

    coin = coin.upper()
    _, binance_symbol = resolve_ccxt_symbols_by_coin(coin)

    end_ts = int(time.time() // 3600 * 3600)
    all_df = fetch_binance_range(binance_symbol, 0, end_ts)
    rows = len(all_df)

    addlog(
        f"[FETCH][ALL] coin={coin} rows={rows}",
        verbose_int=1,
        verbose_state=True,
    )
    return _save_df(coin, all_df)


def fetch_recent(coin: str, hours: int) -> int:
    """Fetch the last ``hours`` candles for ``coin`` and merge with existing data."""

    coin = coin.upper()
    kraken_symbol, _ = resolve_ccxt_symbols_by_coin(coin)

    end_ts = int(time.time() // 3600 * 3600)
    start_ts = end_ts - (hours - 1) * 3600
    recent = fetch_kraken_range(kraken_symbol, start_ts, end_ts)

    start_iso = datetime.fromtimestamp(start_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:00Z")
    end_iso = datetime.fromtimestamp(end_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:00Z")
    addlog(
        f"[FETCH][RECENT] coin={coin} from={start_iso} to={end_iso} rows={len(recent)}",
        verbose_int=1,
        verbose_state=True,
    )

    try:
        existing = load_coin_csv(coin)
    except FileNotFoundError:
        existing = pd.DataFrame(columns=COLUMNS)

    merged = pd.concat([existing, recent], ignore_index=True)
    merged = merged.drop_duplicates(subset="timestamp").sort_values("timestamp")

    return _save_df(coin, merged)


def fetch_recent_coin(coin: str) -> int:
    """Backward-compatible wrapper for fetching the last 720 hours."""

    return fetch_recent(coin, 720)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch candles for a coin")
    parser.add_argument("--coin", required=True, help="Base currency ticker")
    parser.add_argument(
        "--all", action="store_true", help="Fetch full Binance history"
    )
    parser.add_argument(
        "--recent", type=int, help="Fetch recent N hours from Kraken"
    )
    args = parser.parse_args(argv)

    if args.all and args.recent is not None:
        parser.error("--all and --recent cannot be used together")
    if not args.all and args.recent is None:
        parser.error("--all or --recent is required")

    if args.all:
        fetch_all(args.coin)
    else:
        fetch_recent(args.coin, args.recent)


if __name__ == "__main__":
    main()

