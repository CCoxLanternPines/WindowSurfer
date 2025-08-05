from __future__ import annotations

"""Time-aware historical data fetcher."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import List
import sys
import pandas as pd
from systems.utils.resolve_symbol import resolve_ledger_settings
from systems.utils.settings_loader import load_settings

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from systems.utils.time import parse_relative_time
from systems.utils.path import find_project_root

from tqdm import tqdm
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
sys.path.insert(0, str(find_project_root()))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch historical candles")
    parser.add_argument(
        "--tag",
        required=True,
        help="Symbol tag (e.g. SOLDaI)",
    )
    parser.add_argument(
        "--time",
        required=False,
        help="Time window (e.g. 120h)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=1,
        help="Increase verbosity level",
    )
    args = parser.parse_args(argv)

    tag = args.tag.upper()
    time_window = args.time if args.time else "48h"
    verbose = args.verbose

    settings = load_settings()
    ledger_cfg = resolve_ledger_settings(tag, settings)
    kraken_symbol = ledger_cfg["tag"]
    binance_symbol = ledger_cfg["binance_name"]

    start_ts, end_ts = parse_relative_time(time_window)
    out_path = get_raw_path(tag)
    existing = _load_existing(out_path)
    gaps = compute_missing_ranges(existing, start_ts, end_ts, 3_600_000)

    new_frames: List[pd.DataFrame] = []
    added_binance = 0
    added_kraken = 0
    kraken_limited = False

    for gap_start, gap_end in gaps:
        diff_hours = int((gap_end - gap_start) // 3600) + 1
        if diff_hours <= 720:
            addlog(
                f" Fetching from Kraken: {datetime.utcfromtimestamp(gap_start)} to {datetime.utcfromtimestamp(gap_end)}",
                verbose_int=3,
                verbose_state=verbose,
            )
            df = fetch_range("kraken", kraken_symbol, gap_start, gap_end)
            if not df.empty:
                new_frames.append(df)
                added_kraken += len(df)
        else:
            kraken_end = gap_start + 720 * 3600
            addlog(
                f" Fetching from Kraken: {datetime.utcfromtimestamp(gap_start)} to {datetime.utcfromtimestamp(kraken_end)}",
                verbose_int=3,
                verbose_state=verbose,
            )
            df_k = fetch_range("kraken", kraken_symbol, gap_start, kraken_end)
            if not df_k.empty:
                new_frames.append(df_k)
                added_kraken += len(df_k)
                kraken_limited = True
            addlog(
                f" Fetching from Binance: {datetime.utcfromtimestamp(kraken_end + 3600)} to {datetime.utcfromtimestamp(gap_end)}",
                verbose_int=3,
                verbose_state=verbose,
            )
            df_b = fetch_range("binance", binance_symbol, kraken_end + 3600, gap_end)
            if not df_b.empty:
                new_frames.append(df_b)
                added_binance += len(df_b)

    total_rows = _merge_and_save(out_path, existing, new_frames)

    range_str = f"{datetime.fromtimestamp(start_ts, timezone.utc).strftime('%Y-%m-%d')} to {datetime.fromtimestamp(end_ts, timezone.utc).strftime('%Y-%m-%d')}"
    addlog(
        f"ℹ  Target range: {range_str}",
        verbose_int=2,
        verbose_state=verbose,
    )
    existing_rows = existing[
        (existing["timestamp"] >= start_ts) & (existing["timestamp"] <= end_ts)
    ]
    addlog(
        f"ℹ  Existing rows found: {len(existing_rows)}",
        verbose_int=2,
        verbose_state=verbose,
    )

    if added_binance or added_kraken:
        addlog(f" Raw data updated: {out_path}", verbose_int=2, verbose_state=verbose)
        if added_binance:
            addlog(
                f" Added {added_binance} candles from Binance",
                verbose_int=2,
                verbose_state=verbose,
            )
        if added_kraken:
            addlog(
                f" Added {added_kraken} candles from Kraken",
                verbose_int=2,
                verbose_state=verbose,
            )
        if kraken_limited:
            addlog(
                " Could not retrieve full range: Kraken limited to 720 candles",
                verbose_int=2,
                verbose_state=verbose,
            )
    else:
        addlog(
            " Raw data already complete. No candles fetched",
            verbose_int=2,
            verbose_state=verbose,
        )


def fetch_missing_candles(tag: str, relative_window: str = "48h", verbose: int = 1) -> None:
    tag = tag.upper()

    try:
        settings = load_settings()
        ledger_cfg = resolve_ledger_settings(tag, settings)
        kraken_symbol = ledger_cfg["tag"]
        binance_symbol = ledger_cfg["binance_name"]
    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed to resolve symbol '{tag}': {e}")

    start_ts, end_ts = parse_relative_time(relative_window)
    out_path = get_raw_path(tag)
    existing = _load_existing(out_path)
    gaps = compute_missing_ranges(existing, start_ts, end_ts, 3_600_000)

    new_frames: List[pd.DataFrame] = []
    added_binance = 0
    added_kraken = 0

    for gap_start, gap_end in gaps:
        diff_hours = int((gap_end - gap_start) // 3600) + 1
        if diff_hours <= 720:
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
            try:
                df_k = fetch_range("kraken", kraken_symbol, gap_start, kraken_end)
                if not df_k.empty:
                    new_frames.append(df_k)
                    added_kraken += len(df_k)
            except Exception as e:
                addlog(
                    f"[WARN] Kraken fetch error: {e}",
                    verbose_int=2,
                    verbose_state=verbose,
                )
            try:
                df_b = fetch_range("binance", binance_symbol, kraken_end + 3600, gap_end)
                if not df_b.empty:
                    new_frames.append(df_b)
                    added_binance += len(df_b)
            except Exception as e:
                addlog(
                    f"[WARN] Binance fetch error: {e}",
                    verbose_int=2,
                    verbose_state=verbose,
                )

    _merge_and_save(out_path, existing, new_frames)

    addlog(
        f"[SYNC] Added {added_kraken} candles from Kraken, {added_binance} from Binance",
        verbose_int=2,
        verbose_state=verbose,
    )


if __name__ == "__main__":
    main()
