from __future__ import annotations

import argparse
import csv
from typing import List, Tuple

import ccxt

from systems.utils.pairs import resolve_by_tag, raw_path
from systems.utils.time import parse_relative_time

COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def _fetch(exchange: ccxt.Exchange, symbol: str, start_ms: int, end_ms: int) -> List[Tuple[int, float, float, float, float, float]]:
    """Fetch OHLCV data between ``start_ms`` and ``end_ms`` (inclusive)."""
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe="1h", since=start_ms, limit=1000)
    rows: List[Tuple[int, float, float, float, float, float]] = []
    for ts, o, h, l, c, v in ohlcv:
        if ts > end_ms:
            break
        rows.append((ts // 1000, o, h, l, c, v))
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch raw candles by tag")
    parser.add_argument("--tag", required=True, help="Coin tag, e.g. SOL")
    parser.add_argument(
        "--time",
        default="48h",
        help="Relative time window to fetch (e.g. 48h)",
    )
    args = parser.parse_args(argv)

    tag = args.tag.upper()
    start_ts, end_ts = parse_relative_time(args.time)
    start_ms = int(start_ts * 1000)
    end_ms = int(end_ts * 1000)

    pair = resolve_by_tag(tag)
    bn_symbol = pair.get("binance_symbol")
    kr_symbol = pair.get("kraken_symbol")

    frames: List[Tuple[int, float, float, float, float, float]] = []
    if kr_symbol:
        try:
            kr = ccxt.kraken({"enableRateLimit": True})
            frames.extend(_fetch(kr, kr_symbol, start_ms, end_ms))
        except Exception as exc:  # pragma: no cover - network
            print(f"[fetch] kraken error: {exc}")
    if bn_symbol:
        try:
            bn = ccxt.binance({"enableRateLimit": True})
            frames.extend(_fetch(bn, bn_symbol, start_ms, end_ms))
        except Exception as exc:  # pragma: no cover - network
            print(f"[fetch] binance error: {exc}")

    dedup = {ts: (ts, o, h, l, c, v) for ts, o, h, l, c, v in frames}
    ordered = [dedup[k] for k in sorted(dedup)]

    path = raw_path(tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        writer.writerows(ordered)

    print(f"[fetch] wrote {len(ordered)} rows to {path}")


if __name__ == "__main__":
    main()
