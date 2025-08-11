from __future__ import annotations

"""CLI entry for deterministic, gapless historical fetches."""

from datetime import datetime, timezone
from typing import Optional

from systems.utils.cli import build_parser
from systems.utils.config import load_settings, resolve_path
from systems.scripts.fetch_core import (
    FetchAbort,
    _fetch_binance,
    _fetch_kraken,
    canonicalize,
    get_raw_path,
    reindex_hourly,
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
    tag = cfg.get("tag")
    binance_symbol = cfg.get("binance_name")
    kraken_symbol = cfg.get("kraken_name")

    candles = parse_time_to_candles(args.time)
    now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    end_ms = floor_to_top_of_hour(now_ms)
    start_ms = end_ms - candles * 3_600_000

    start_iso = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).isoformat()
    end_iso = datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc).isoformat()

    print(
        f"[FETCH] ledger={ledger_name} time={args.time} candles={candles} "
        f"binance={binance_symbol} kraken={kraken_symbol} range={start_iso}→{end_iso}"
    )

    out_path = get_raw_path(tag)
    root = resolve_path("")
    print(f"[FETCH][PATH] root={root} out={out_path}")

    try:
        if candles > 720:
            if out_path.exists():
                print(f"[REFRESH] deleting cache at {out_path}")
                out_path.unlink()
            df_b = _fetch_binance(binance_symbol, start_ms, end_ms)
            df_b = canonicalize(df_b)
            if df_b.empty:
                raise FetchAbort(
                    f"Binance returned 0 candles for span. symbol={binance_symbol} "
                    f"range={start_iso}→{end_iso}"
                )
            df_full, gaps = reindex_hourly(df_b, start_ms, end_ms)
            if gaps:
                raise FetchAbort(
                    f"Full refresh not gapless. symbol={binance_symbol} gaps={gaps} "
                    "(Policy: >720 → no healing. Use a symbol with full history or shorten --time.)"
                )
        else:
            df_k = _fetch_kraken(kraken_symbol, start_ms, end_ms)
            df_k = canonicalize(df_k)
            df_full, gaps = reindex_hourly(df_k, start_ms, end_ms)
            if gaps:
                print(f"[HEAL][≤720] gaps={len(gaps)} via Binance (same span)")
                df_b = _fetch_binance(binance_symbol, start_ms, end_ms)
                df_b = canonicalize(df_b)
                df_full_idx = df_full.set_index("ts")
                df_b_idx = df_b.set_index("ts")
                df_full = df_full_idx.combine_first(df_b_idx).reset_index()
                df_full, gaps = reindex_hourly(df_full, start_ms, end_ms)
                if gaps:
                    raise FetchAbort(
                        "Unhealable gaps in ≤720 span. symbols: "
                        f"kraken={kraken_symbol} binance={binance_symbol} details={gaps}"
                    )
        df_full.to_parquet(out_path, index=False)
        print(
            f"[FETCH] gapless \u2713 rows={len(df_full)} span={start_iso}→{end_iso}"
        )
    except FetchAbort as exc:
        print(f"[ABORT][FETCH] {exc}")


if __name__ == "__main__":
    main()
