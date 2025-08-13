from __future__ import annotations

"""Fetch helper exposing a recent-only path."""

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
    fetch_last_720_kraken,
    get_coin_raw_path,
    heal_recent,
)


def fetch_recent_coin(coin: str) -> int:
    """Fetch and heal the last 720 candles for ``coin``."""

    coin = coin.upper()
    kraken_symbol, _ = resolve_ccxt_symbols_by_coin(coin)

    end_ts = int(time.time() // 3600 * 3600)
    recent = fetch_last_720_kraken(kraken_symbol, end_ts)
    rows = len(recent)

    start_ts = end_ts - 719 * 3600
    start_iso = datetime.fromtimestamp(start_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:00Z")
    end_iso = datetime.fromtimestamp(end_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:00Z")
    addlog(
        f"[FETCH][RECENT] coin={coin} from={start_iso} to={end_iso} rows={rows}",
        verbose_int=1,
        verbose_state=True,
    )

    try:
        existing = load_coin_csv(coin)
    except FileNotFoundError:
        existing = pd.DataFrame(columns=COLUMNS)

    merged, appended, dedup, gaps = heal_recent(existing, recent)
    addlog(
        f"[HEAL] appended={appended} dedup={dedup} total={len(merged)}",
        verbose_int=1,
        verbose_state=True,
    )
    addlog(
        f"[SYNC] Remaining recent-window gaps: {gaps} hour(s)",
        verbose_int=1,
        verbose_state=True,
    )

    path = get_coin_raw_path(coin)
    tmp = path.with_suffix(".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(tmp, index=False)
    tmp.replace(path)
    return len(merged)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch recent candles for a coin")
    parser.add_argument("--recent", action="store_true", help="Fetch last 720h from Kraken")
    parser.add_argument("--coin", required=True, help="Base currency ticker")
    args = parser.parse_args(argv)

    if not args.recent:
        parser.error("--recent is required")

    fetch_recent_coin(args.coin)


if __name__ == "__main__":
    main()

