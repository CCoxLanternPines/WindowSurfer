from __future__ import annotations

"""Time-aware historical data fetcher."""

from datetime import datetime, timezone
from pathlib import Path
from typing import List
import sys
import pandas as pd
from systems.utils.config import load_ledger_config, resolve_path
from systems.utils.cli import build_parser
from systems.utils.resolve_symbol import (
    refresh_pair_cache,
    load_pair_cache,
    resolve_ccxt_symbols,
)

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from systems.utils.time import parse_relative_time

from systems.utils.addlog import addlog
from systems.scripts.fetch_core import (
    _load_existing,
    _merge_and_save,
    get_raw_path,
    COLUMNS,
    compute_missing_ranges,
    fetch_range,
)

# **Inject** the project root so “utils” is importable:
sys.path.insert(0, str(resolve_path("")))


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    parser.add_argument(
        "--time",
        required=False,
        help="Time window (e.g. 120h)",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Refresh exchange pair cache before running",
    )
    args = parser.parse_args(argv)
    if not args.ledger:
        parser.error("--ledger is required")

    time_window = args.time if args.time else "48h"
    verbose = args.verbose

    ledger_cfg = load_ledger_config(args.ledger)
    tag = ledger_cfg["tag"].upper()
    if getattr(args, "cache", False):
        refresh_pair_cache(verbose)
    cache = load_pair_cache()
    coin = ledger_cfg.get("coin")
    fiat = ledger_cfg.get("fiat")
    if coin and fiat:
        syms = resolve_ccxt_symbols(coin, fiat, cache, verbose)
        kraken_symbol = syms["kraken_name"]
        binance_symbol = syms["binance_name"]
    else:
        kraken_symbol = ledger_cfg.get("kraken_name")
        binance_symbol = ledger_cfg.get("binance_name")
        if not kraken_symbol or not binance_symbol:
            addlog(
                f"[ERROR] Missing kraken_name/binance_name in settings for ledger {args.ledger}",
                verbose_int=1,
                verbose_state=True,
            )
            sys.exit(1)

    start_ts, end_ts = parse_relative_time(time_window)
    start_ts = int(start_ts // 3600 * 3600)
    end_ts = int(end_ts // 3600 * 3600)
    interval_ms = 3_600_000
    out_path = get_raw_path(tag)
    existing = _load_existing(out_path)
    gaps = compute_missing_ranges(existing, start_ts, end_ts, interval_ms)

    new_frames: List[pd.DataFrame] = []
    added_binance = 0
    added_kraken = 0
    kraken_limited = False

    for gap_start, gap_end in gaps:
        diff_hours = int((gap_end - gap_start) // 3600) + 1
        iso = lambda ts: datetime.utcfromtimestamp(ts).isoformat()
        if diff_hours <= 720:
            addlog(
                f"[FETCH] Kraken {kraken_symbol} {iso(gap_start)} → {iso(gap_end)} (≤720h)",
                verbose_int=2,
                verbose_state=verbose,
            )
            df = fetch_range("kraken", kraken_symbol, gap_start, gap_end)
            if not df.empty:
                new_frames.append(df)
                added_kraken += len(df)
        else:
            kraken_end = gap_start + 720 * 3600
            addlog(
                f"[FETCH] Kraken {kraken_symbol} {iso(gap_start)} → {iso(kraken_end)} (≤720h)",
                verbose_int=2,
                verbose_state=verbose,
            )
            df_k = fetch_range("kraken", kraken_symbol, gap_start, kraken_end)
            if not df_k.empty:
                new_frames.append(df_k)
                added_kraken += len(df_k)
                kraken_limited = True
            binance_start = kraken_end + 3600
            addlog(
                f"[FETCH] Binance {binance_symbol} {iso(binance_start)} → {iso(gap_end)} (tail)",
                verbose_int=2,
                verbose_state=verbose,
            )
            df_b = fetch_range("binance", binance_symbol, binance_start, gap_end)
            if not df_b.empty:
                new_frames.append(df_b)
                added_binance += len(df_b)

    total_rows = _merge_and_save(out_path, existing, new_frames)
    post_df = _load_existing(out_path)
    post_gaps = compute_missing_ranges(post_df, start_ts, end_ts, interval_ms)
    if post_gaps:
        missing_hours = sum(int((e - s) // 3600) + 1 for s, e in post_gaps)
        addlog(
            f"[WARN] Post-merge gaps detected: {missing_hours} hour(s) missing",
            verbose_state=True,
        )
    else:
        addlog(
            "[SYNC] No post-merge gaps remain",
            verbose_int=2,
            verbose_state=verbose,
        )

    addlog(
        f"[SYNC] Total rows in file: {total_rows}",
        verbose_int=2,
        verbose_state=verbose,
    )

    addlog(
        f"[SYNC] Added {added_kraken} candles from Kraken, {added_binance} from Binance",
        verbose_int=2,
        verbose_state=verbose,
    )
    if kraken_limited:
        addlog(
            "[INFO] Could not retrieve full range: Kraken limited to 720 candles",
            verbose_int=2,
            verbose_state=verbose,
        )
    if not added_binance and not added_kraken:
        addlog(
            " Raw data already complete. No candles fetched",
            verbose_int=2,
            verbose_state=verbose,
        )


def fetch_missing_candles(
    ledger: str,
    relative_window: str = "48h",
    verbose: int = 1,
    refresh_cache: bool = False,
) -> None:
    ledger_cfg = load_ledger_config(ledger)
    tag = ledger_cfg["tag"].upper()
    if refresh_cache:
        refresh_pair_cache(verbose)
    cache = load_pair_cache()
    coin = ledger_cfg.get("coin")
    fiat = ledger_cfg.get("fiat")
    if coin and fiat:
        syms = resolve_ccxt_symbols(coin, fiat, cache, verbose)
        kraken_symbol = syms["kraken_name"]
        binance_symbol = syms["binance_name"]
    else:
        kraken_symbol = ledger_cfg.get("kraken_name")
        binance_symbol = ledger_cfg.get("binance_name")
        if not kraken_symbol or not binance_symbol:
            addlog(
                f"[ERROR] Missing kraken_name/binance_name in settings for ledger {ledger}",
                verbose_int=1,
                verbose_state=True,
            )
            raise RuntimeError("Missing exchange symbols")

    start_ts, end_ts = parse_relative_time(relative_window)
    start_ts = int(start_ts // 3600 * 3600)
    end_ts = int(end_ts // 3600 * 3600)
    interval_ms = 3_600_000
    out_path = get_raw_path(tag)
    existing = _load_existing(out_path)
    gaps = compute_missing_ranges(existing, start_ts, end_ts, interval_ms)

    new_frames: List[pd.DataFrame] = []
    added_binance = 0
    added_kraken = 0
    kraken_limited = False

    for gap_start, gap_end in gaps:
        diff_hours = int((gap_end - gap_start) // 3600) + 1
        iso = lambda ts: datetime.utcfromtimestamp(ts).isoformat()
        if diff_hours <= 720:
            addlog(
                f"[FETCH] Kraken {kraken_symbol} {iso(gap_start)} → {iso(gap_end)} (≤720h)",
                verbose_int=2,
                verbose_state=verbose,
            )
            try:
                df = fetch_range("kraken", kraken_symbol, gap_start, gap_end)
                if not df.empty:
                    new_frames.append(df)
                    added_kraken += len(df)
            except Exception as e:
                addlog(
                    f"[WARN] Kraken fetch error: {e}",
                    verbose_int=2,
                    verbose_state=verbose,
                )
        else:
            kraken_end = gap_start + 720 * 3600
            addlog(
                f"[FETCH] Kraken {kraken_symbol} {iso(gap_start)} → {iso(kraken_end)} (≤720h)",
                verbose_int=2,
                verbose_state=verbose,
            )
            try:
                df_k = fetch_range("kraken", kraken_symbol, gap_start, kraken_end)
                if not df_k.empty:
                    new_frames.append(df_k)
                    added_kraken += len(df_k)
                    kraken_limited = True
            except Exception as e:
                addlog(
                    f"[WARN] Kraken fetch error: {e}",
                    verbose_int=2,
                    verbose_state=verbose,
                )
            binance_start = kraken_end + 3600
            addlog(
                f"[FETCH] Binance {binance_symbol} {iso(binance_start)} → {iso(gap_end)} (tail)",
                verbose_int=2,
                verbose_state=verbose,
            )
            try:
                df_b = fetch_range("binance", binance_symbol, binance_start, gap_end)
                if not df_b.empty:
                    new_frames.append(df_b)
                    added_binance += len(df_b)
            except Exception as e:
                addlog(
                    f"[WARN] Binance fetch error: {e}",
                    verbose_int=2,
                    verbose_state=verbose,
                )

    total_rows = _merge_and_save(out_path, existing, new_frames)
    post_df = _load_existing(out_path)
    post_gaps = compute_missing_ranges(post_df, start_ts, end_ts, interval_ms)
    if post_gaps:
        missing_hours = sum(int((e - s) // 3600) + 1 for s, e in post_gaps)
        addlog(
            f"[WARN] Post-merge gaps detected: {missing_hours} hour(s) missing",
            verbose_state=True,
        )
    else:
        addlog(
            "[SYNC] No post-merge gaps remain",
            verbose_int=2,
            verbose_state=verbose,
        )

    addlog(
        f"[SYNC] Total rows in file: {total_rows}",
        verbose_int=2,
        verbose_state=verbose,
    )

    addlog(
        f"[SYNC] Added {added_kraken} candles from Kraken, {added_binance} from Binance",
        verbose_int=2,
        verbose_state=verbose,
    )
    if kraken_limited:
        addlog(
            "[INFO] Could not retrieve full range: Kraken limited to 720 candles",
            verbose_int=2,
            verbose_state=verbose,
        )
    if not added_binance and not added_kraken:
        addlog(
            " Raw data already complete. No candles fetched",
            verbose_int=2,
            verbose_state=verbose,
        )


if __name__ == "__main__":
    main()
