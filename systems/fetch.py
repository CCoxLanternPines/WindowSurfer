from __future__ import annotations

"""Utilities for fetching and storing market data."""

import argparse
import sys
import time
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from systems.utils.addlog import addlog
from systems.utils.config import load_settings
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


def _get_symbols(coin: str) -> tuple[str, str]:
    settings = load_settings()
    try:
        entry = settings["ledger_settings"][coin]
        return entry["kraken_name"], entry["binance_name"]
    except KeyError as exc:
        raise KeyError(f"Missing exchange symbol mapping for {coin}") from exc


def fetch_all(coin: str) -> int:
    """Fetch full Binance history for ``coin`` and write to disk."""

    coin = coin.upper()
    _, binance_symbol = _get_symbols(coin)

    end_ts = int(time.time() // 3600 * 3600)
    all_df = fetch_binance_range(binance_symbol, 0, end_ts)

    addlog(
        f"[FETCH][ALL] {coin} via Binance symbol={binance_symbol}",
        verbose_int=1,
        verbose_state=True,
    )
    raw_path = get_coin_raw_path(coin)
    raw_path.unlink(missing_ok=True)
    return _save_df(coin, all_df)


def fetch_recent(coin: str, hours: int) -> int:
    """Fetch the last ``hours`` candles for ``coin`` and merge with existing data."""

    coin = coin.upper()
    kraken_symbol, binance_symbol = _get_symbols(coin)

    end_ts = int(time.time() // 3600 * 3600)
    frames = []

    if hours <= 720:
        start_ts = end_ts - (hours - 1) * 3600
        frames.append(fetch_kraken_range(kraken_symbol, start_ts, end_ts))
    else:
        kraken_start = end_ts - 719 * 3600
        bin_start = end_ts - (hours - 1) * 3600
        bin_end = kraken_start - 3600
        frames.append(fetch_binance_range(binance_symbol, bin_start, bin_end))
        frames.append(fetch_kraken_range(kraken_symbol, kraken_start, end_ts))

    try:
        existing = load_coin_csv(coin)
    except FileNotFoundError:
        existing = pd.DataFrame(columns=COLUMNS)

    merged = pd.concat([existing] + frames, ignore_index=True)
    merged = merged.drop_duplicates(subset="timestamp").sort_values("timestamp")

    addlog(
        f"[FETCH][RECENT] {coin} hours={hours} kraken={kraken_symbol} binance={binance_symbol}",
        verbose_int=1,
        verbose_state=True,
    )

    return _save_df(coin, merged)


def fetch_recent_coin(coin: str) -> int:
    """Backward-compatible wrapper for fetching the last 720 hours."""

    return fetch_recent(coin, 720)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch candles for a coin")
    parser.add_argument("--coin", required=True, help="Base currency ticker")
    parser.add_argument(
        "--all", action="store_true", help="Fetch full Binance history",
    )
    parser.add_argument(
        "--recent", type=int, help="Fetch recent N hours from Kraken",
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
