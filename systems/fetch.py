from __future__ import annotations

"""CLI entry for deterministic, gapless historical fetches."""

from datetime import datetime, timezone
from typing import Optional

from systems.utils.cli import build_parser
from systems.utils.config import load_settings
from systems.scripts.fetch_core import (
    FetchAbort,
    get_gapless_1h_for_span,
    get_raw_path,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def parse_time_to_candles(spec: str) -> int:
    spec = spec.strip().lower()
    unit = spec[-1]
    num = int(spec[:-1])
    factors = {
        "h": 1,
        "d": 24,
        "w": 24 * 7,
        "m": 24 * 30,
        "y": 24 * 365,
    }
    if unit not in factors:
        raise ValueError("time spec must end with h,d,w,m,y")
    return num * factors[unit]


def floor_to_top_of_hour(ts_ms: int) -> int:
    return (ts_ms // 3_600_000) * 3_600_000


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    parser.add_argument("--time", required=True, help="Fetch window, e.g. 90d")
    args = parser.parse_args(argv)

    settings = load_settings()
    ledger_name = args.ledger or "default"
    ledgers = settings.get("ledger_settings", {})
    cfg = ledgers.get(ledger_name, ledgers.get("default", {}))
    binance_symbol = cfg.get("binance_name")
    kraken_symbol = cfg.get("kraken_name")

    print(
        f"[FETCH] ledger={ledger_name} time={args.time} "
        f"binance={binance_symbol} kraken={kraken_symbol}"
    )

    candles = parse_time_to_candles(args.time)
    now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    end_ms = floor_to_top_of_hour(now_ms)
    start_ms = end_ms - candles * 3_600_000

    try:
        df = get_gapless_1h_for_span(cfg, start_ms, end_ms)
    except FetchAbort as exc:
        print(f"[ABORT][FETCH] {exc}")
        return

    out_path = get_raw_path(cfg["tag"].upper())
    df.to_parquet(out_path, index=False)

    start_iso = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).isoformat()
    end_iso = datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc).isoformat()
    print(
        f"[FETCH] gapless \u2713 rows={len(df)} span=[{start_iso}, {end_iso}]"
    )


if __name__ == "__main__":
    main()
