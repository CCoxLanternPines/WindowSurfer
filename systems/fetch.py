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
    _fetch_kraken,
    _fetch_binance,
    _load_existing,
    _merge_and_save,
    get_raw_path,
    COLUMNS,
)

# **Inject** the project root so “utils” is importable:
sys.path.insert(0, str(find_project_root()))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch historical candles")
    parser.add_argument(
        "--ledger",
        required=True,
        help="Ledger name (e.g. Kris_Ledger)",
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

    ledger_name = args.ledger
    time_window = args.time if args.time else "48h"
    verbose = args.verbose

    settings = load_settings()
    try:
        ledger_cfg = resolve_ledger_settings(ledger_name, settings)
    except Exception as e:
        raise RuntimeError(f"Ledger '{ledger_name}' not found: {e}")
    tag = ledger_cfg["tag"]
    kraken_symbol = ledger_cfg["kraken_name"]
    binance_symbol = ledger_cfg.get("binance_name")
    log_prefix = f"[FETCH] {ledger_name} | {tag}"

    def log(message: str, **kwargs) -> None:
        addlog(f"{log_prefix} {message}", **kwargs)

    start_ts, end_ts = parse_relative_time(time_window)
    out_path = get_raw_path(tag)
    existing = _load_existing(out_path)

    existing_rows = existing[(existing["timestamp"] >= start_ts) & (existing["timestamp"] <= end_ts)]
    earliest = existing_rows["timestamp"].min() if not existing_rows.empty else float("nan")
    latest = existing_rows["timestamp"].max() if not existing_rows.empty else float("nan")

    early_gap = None
    late_gap = None
    if existing_rows.empty:
        early_gap = (start_ts, end_ts)
    else:
        if earliest > start_ts or pd.isna(earliest):
            early_gap = (start_ts, min(earliest - 3600, end_ts)) if not pd.isna(earliest) else (start_ts, end_ts)
        if latest < end_ts or pd.isna(latest):
            late_gap = (max(latest + 3600, start_ts) if not pd.isna(latest) else start_ts, end_ts)

    new_frames: List[pd.DataFrame] = []
    added_binance = 0
    added_kraken = 0
    kraken_limited = False


    def fetch_and_store_combined(gap):
        nonlocal added_kraken, added_binance, kraken_limited
        if not gap or gap[0] > gap[1]:
            return

        start_ms = int(gap[0] * 1000)
        end_ms = int(gap[1] * 1000)
        diff_hours = int((end_ms - start_ms) // 3600000) + 1

        if diff_hours <= 120:
            log(
                f"Fetching from Kraken: {datetime.utcfromtimestamp(start_ms/1000)} to {datetime.utcfromtimestamp(end_ms/1000)}",
                verbose_int=3,
                verbose_state=verbose,
            )
            try:
                rows = _fetch_kraken(kraken_symbol, start_ms, end_ms)
            except Exception as e:
                log(
                    f"Error fetching from Kraken: {e}",
                    verbose_int=3,
                    verbose_state=verbose,
                )
                rows = []
            log(
                f"{len(rows)} rows fetched from Kraken",
                verbose_int=3,
                verbose_state=verbose,
            )
            if rows:
                df = pd.DataFrame(rows, columns=COLUMNS)
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).astype("int64") // 1_000_000_000
                new_frames.append(df)
                added_kraken += len(df)
        else:
            # Split the request into two fetches:
            # Kraken gets first 120h, Binance gets the rest
            kraken_end_ms = start_ms + 120 * 3600000
            log(
                f"Fetching from Kraken: {datetime.utcfromtimestamp(start_ms/1000)} to {datetime.utcfromtimestamp(kraken_end_ms/1000)}",
                verbose_int=3,
                verbose_state=verbose,
            )
            try:
                kraken_rows = _fetch_kraken(kraken_symbol, start_ms, kraken_end_ms)
                kraken_limited = True
            except Exception as e:
                log(
                    f"Error fetching from Kraken: {e}",
                    verbose_int=3,
                    verbose_state=verbose,
                )
                kraken_rows = []
            log(
                f"{len(kraken_rows)} rows fetched from Kraken",
                verbose_int=3,
                verbose_state=verbose,
            )

            if kraken_rows:
                df_k = pd.DataFrame(kraken_rows, columns=COLUMNS)
                df_k["timestamp"] = pd.to_datetime(df_k["timestamp"], unit="ms", utc=True).astype("int64") // 1_000_000_000
                new_frames.append(df_k)
                added_kraken += len(df_k)

            log(
                f"Fetching from Binance: {datetime.utcfromtimestamp((kraken_end_ms+3600000)/1000)} to {datetime.utcfromtimestamp(end_ms/1000)}",
                verbose_int=3,
                verbose_state=verbose,
            )
            try:
                binance_rows = _fetch_binance(binance_symbol, kraken_end_ms + 3600000, end_ms)
            except Exception as e:
                log(
                    f"Error fetching from Binance: {e}",
                    verbose_int=3,
                    verbose_state=verbose,
                )
                binance_rows = []
            log(
                f"{len(binance_rows)} rows fetched from Binance",
                verbose_int=3,
                verbose_state=verbose,
            )

            if binance_rows:
                df_b = pd.DataFrame(binance_rows, columns=COLUMNS)
                df_b["timestamp"] = pd.to_datetime(df_b["timestamp"], unit="ms", utc=True).astype("int64") // 1_000_000_000
                new_frames.append(df_b)
                added_binance += len(df_b)


    # Execute the gap fills
    fetch_and_store_combined(early_gap)
    fetch_and_store_combined(late_gap)

    total_rows = _merge_and_save(out_path, existing, new_frames)

    range_str = f"{datetime.fromtimestamp(start_ts, timezone.utc).strftime('%Y-%m-%d')} to {datetime.fromtimestamp(end_ts, timezone.utc).strftime('%Y-%m-%d')}"
    log(
        f"ℹ  Target range: {range_str}",
        verbose_int=2,
        verbose_state=verbose,
    )
    log(
        f"ℹ  Existing rows found: {len(existing_rows)}",
        verbose_int=2,
        verbose_state=verbose,
    )

    if added_binance or added_kraken:
        log(f"Raw data updated: {out_path}", verbose_int=2, verbose_state=verbose)
        if added_binance:
            log(
                f"Added {added_binance} candles from Binance",
                verbose_int=2,
                verbose_state=verbose,
            )
        if added_kraken:
            log(
                f"Added {added_kraken} candles from Kraken",
                verbose_int=2,
                verbose_state=verbose,
            )
        if kraken_limited:
            log(
                "Could not retrieve full range: Kraken limited to 120 candles",
                verbose_int=2,
                verbose_state=verbose,
            )
    else:
        log(
            "Raw data already complete. No candles fetched",
            verbose_int=2,
            verbose_state=verbose,
        )


def fetch_missing_candles(
    ledger_name: str, relative_window: str = "48h", verbose: int = 1
) -> None:
    try:
        settings = load_settings()
        ledger_cfg = resolve_ledger_settings(ledger_name, settings)
        tag = ledger_cfg["tag"]
        kraken_symbol = ledger_cfg["kraken_name"]
        binance_symbol = ledger_cfg.get("binance_name")
    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed to resolve ledger '{ledger_name}': {e}")

    log_prefix = f"[FETCH] {ledger_name} | {tag}"

    def log(message: str, **kwargs) -> None:
        addlog(f"{log_prefix} {message}", **kwargs)

    start_ts, end_ts = parse_relative_time(relative_window)
    out_path = get_raw_path(tag)
    existing = _load_existing(out_path)

    existing_rows = existing[(existing["timestamp"] >= start_ts) & (existing["timestamp"] <= end_ts)]
    earliest = existing_rows["timestamp"].min() if not existing_rows.empty else float("nan")
    latest = existing_rows["timestamp"].max() if not existing_rows.empty else float("nan")

    early_gap = None
    late_gap = None
    if existing_rows.empty:
        early_gap = (start_ts, end_ts)
    else:
        if earliest > start_ts or pd.isna(earliest):
            early_gap = (start_ts, min(earliest - 3600, end_ts)) if not pd.isna(earliest) else (start_ts, end_ts)
        if latest < end_ts or pd.isna(latest):
            late_gap = (max(latest + 3600, start_ts) if not pd.isna(latest) else start_ts, end_ts)

    new_frames: List[pd.DataFrame] = []
    added_binance = 0
    added_kraken = 0

    def fetch_and_store_combined(gap):
        nonlocal added_kraken, added_binance
        if not gap or gap[0] > gap[1]:
            return

        start_ms = int(gap[0] * 1000)
        end_ms = int(gap[1] * 1000)
        diff_hours = int((end_ms - start_ms) // 3600000) + 1

        if diff_hours <= 120:
            try:
                rows = _fetch_kraken(kraken_symbol, start_ms, end_ms)
                if rows:
                    df = pd.DataFrame(rows, columns=COLUMNS)
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).astype("int64") // 1_000_000_000
                    new_frames.append(df)
                    added_kraken += len(df)
            except Exception as e:
                log(
                    f"[WARN] Kraken fetch error: {e}",
                    verbose_int=2,
                    verbose_state=verbose,
                )
        else:
            kraken_end_ms = start_ms + 120 * 3600000
            try:
                rows = _fetch_kraken(kraken_symbol, start_ms, kraken_end_ms)
                if rows:
                    df_k = pd.DataFrame(rows, columns=COLUMNS)
                    df_k["timestamp"] = pd.to_datetime(df_k["timestamp"], unit="ms", utc=True).astype("int64") // 1_000_000_000
                    new_frames.append(df_k)
                    added_kraken += len(df_k)
            except Exception as e:
                log(
                    f"[WARN] Kraken fetch error: {e}",
                    verbose_int=2,
                    verbose_state=verbose,
                )

            try:
                rows = _fetch_binance(binance_symbol, kraken_end_ms + 3600000, end_ms)
                if rows:
                    df_b = pd.DataFrame(rows, columns=COLUMNS)
                    df_b["timestamp"] = pd.to_datetime(df_b["timestamp"], unit="ms", utc=True).astype("int64") // 1_000_000_000
                    new_frames.append(df_b)
                    added_binance += len(df_b)
            except Exception as e:
                log(
                    f"[WARN] Binance fetch error: {e}",
                    verbose_int=2,
                    verbose_state=verbose,
                )

    fetch_and_store_combined(early_gap)
    fetch_and_store_combined(late_gap)

    _merge_and_save(out_path, existing, new_frames)

    log(
        f"[SYNC] Added {added_kraken} candles from Kraken, {added_binance} from Binance",
        verbose_int=2,
        verbose_state=verbose,
    )


if __name__ == "__main__":
    main()
