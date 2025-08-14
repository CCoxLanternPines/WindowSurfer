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
from systems.scripts.fetch_core import (
    fetch_binance_range,
    fetch_kraken_range,
    get_coin_raw_path,
)


def _write_clean_csv(csv_path: str, frames):
    """Merge frames, normalize timestamp, drop open candle, sort ascending, dedupe on timestamp."""
    now_ms = int(time.time() * 1000)
    df = pd.concat(frames, ignore_index=True)

    # Normalize timestamp column name + type
    ts_col = None
    for c in df.columns:
        lc = str(c).lower()
        if lc in ("timestamp", "time", "date"):
            ts_col = c
            break
    if ts_col is None:
        raise ValueError(f"No timestamp-like column in fetched frames for {csv_path}")

    if ts_col != "timestamp":
        df.rename(columns={ts_col: "timestamp"}, inplace=True)

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    # Drop the currently building 1h candle (anything within the last hour)
    df = df[df["timestamp"] <= now_ms - 3600000]

    # Sort ascending + dedupe on timestamp
    before = len(df)
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    removed = before - len(df)

    df.to_csv(csv_path, index=False)
    print(f"[FETCH][WRITE] path={csv_path} rows={len(df)} dups_removed={removed}")


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
    csv_path = get_coin_raw_path(coin)
    _write_clean_csv(csv_path, [all_df])
    return rows


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

    csv_path = get_coin_raw_path(coin)
    existing = pd.read_csv(csv_path)
    _write_clean_csv(csv_path, [existing, recent])
    return len(recent)


def fetch_recent_coin(coin: str) -> int:
    """Backward-compatible wrapper for fetching the last 720 hours."""

    return fetch_recent(coin, 720)


def run_fetch(coin: str | None, *, fetch_all: bool, recent: int | None) -> None:
    """CLI helper for fetching candle data.

    Parameters mirror the command line flags and validation mirrors the
    previous inline logic in ``bot.py``.
    """

    if not coin:
        addlog(
            "Error: --coin is required for fetch mode",
            verbose_int=1,
            verbose_state=True,
        )
        raise SystemExit(1)

    if fetch_all and recent is not None:
        addlog(
            "Error: --all and --recent are mutually exclusive",
            verbose_int=1,
            verbose_state=True,
        )
        raise SystemExit(1)

    if not fetch_all and recent is None:
        addlog(
            "Error: either --all or --recent is required",
            verbose_int=1,
            verbose_state=True,
        )
        raise SystemExit(1)

    if fetch_all:
        addlog(
            f"[BOT][FETCH][ALL] coin={coin} → full Binance history",
            verbose_int=1,
            verbose_state=True,
        )
        fetch_all(coin)
    else:
        addlog(
            f"[BOT][FETCH][RECENT] coin={coin} hours={recent}",
            verbose_int=1,
            verbose_state=True,
        )
        fetch_recent(coin, recent)


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

