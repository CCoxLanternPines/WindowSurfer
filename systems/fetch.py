from __future__ import annotations

"""Time-aware historical data fetcher."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List
import sys
import pandas as pd
from systems.utils.resolve_symbol import resolve_symbol

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from systems.utils.time import parse_relative_time
from systems.utils.path import find_project_root

from tqdm import tqdm
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

from systems.utils.resolve_symbol import resolve_symbol

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch historical candles")
    parser.add_argument(
        "--tag",
        required=True,
        help="Market tag in symbol_settings.json",
    )
    
    symbols = resolve_symbol(tag)
    kraken_symbol = symbols["kraken"]
    binance_symbol = symbols["binance"]
    
    parser.add_argument("--time", required=True, help="Relative window like 10d or 2y")
    args = parser.parse_args(argv)

    tag = args.tag.upper()

    try:
        symbols = resolve_symbol(tag)
        kraken_symbol = symbols["kraken"]
        binance_symbol = symbols["binance"]
    except Exception as e:
        print(f"[ERROR] Failed to resolve symbol for tag '{tag}': {e}")
        return


    start_ts, end_ts = parse_relative_time(args.time)
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


    def fetch_and_store(gap):
        nonlocal added_kraken, kraken_limited
        if not gap or gap[0] > gap[1]:
            return

        start_ms = int(gap[0] * 1000)
        original_end_ms = int(gap[1] * 1000)
        diff_hours = int((original_end_ms - start_ms) // 3600000) + 1

        if diff_hours > 720:
            kraken_limited = True
            end_ms = start_ms + 720 * 3600000  # truncate to 720h max
        else:
            end_ms = original_end_ms

        print(f" Fetching from Kraken: {datetime.utcfromtimestamp(start_ms/1000)} to {datetime.utcfromtimestamp(end_ms/1000)}")

        try:
            rows = _fetch_kraken(kraken_symbol, start_ms, end_ms)
        except Exception as e:
            print(f" Error fetching from Kraken: {e}")
            return

        print(f" {len(rows)} rows fetched from Kraken")

        df = pd.DataFrame(rows, columns=COLUMNS)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).astype("int64") // 1_000_000_000
            new_frames.append(df)
            added_kraken += len(df)

    
    def fetch_and_store_combined(gap):
        nonlocal added_kraken, added_binance, kraken_limited
        if not gap or gap[0] > gap[1]:
            return

        start_ms = int(gap[0] * 1000)
        end_ms = int(gap[1] * 1000)
        diff_hours = int((end_ms - start_ms) // 3600000) + 1

        if diff_hours <= 720:
            print(f" Fetching from Kraken: {datetime.utcfromtimestamp(start_ms/1000)} to {datetime.utcfromtimestamp(end_ms/1000)}")
            try:
                rows = _fetch_kraken(kraken_symbol, start_ms, end_ms)
            except Exception as e:
                print(f" Error fetching from Kraken: {e}")
                rows = []
            print(f" {len(rows)} rows fetched from Kraken")
            if rows:
                df = pd.DataFrame(rows, columns=COLUMNS)
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).astype("int64") // 1_000_000_000
                new_frames.append(df)
                added_kraken += len(df)
        else:
            # Split the request into two fetches:
            # Kraken gets first 720h, Binance gets the rest
            kraken_end_ms = start_ms + 720 * 3600000
            print(f" Fetching from Kraken: {datetime.utcfromtimestamp(start_ms/1000)} to {datetime.utcfromtimestamp(kraken_end_ms/1000)}")
            try:
                kraken_rows = _fetch_kraken(kraken_symbol, start_ms, kraken_end_ms)
                kraken_limited = True
            except Exception as e:
                print(f" Error fetching from Kraken: {e}")
                kraken_rows = []
            print(f" {len(kraken_rows)} rows fetched from Kraken")

            if kraken_rows:
                df_k = pd.DataFrame(kraken_rows, columns=COLUMNS)
                df_k["timestamp"] = pd.to_datetime(df_k["timestamp"], unit="ms", utc=True).astype("int64") // 1_000_000_000
                new_frames.append(df_k)
                added_kraken += len(df_k)

            print(f" Fetching from Binance: {datetime.utcfromtimestamp((kraken_end_ms+3600000)/1000)} to {datetime.utcfromtimestamp(end_ms/1000)}")
            try:
                binance_rows = _fetch_binance(binance_symbol, kraken_end_ms + 3600000, end_ms)
            except Exception as e:
                print(f" Error fetching from Binance: {e}")
                binance_rows = []
            print(f" {len(binance_rows)} rows fetched from Binance")

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
    print(f"ℹ  Target range: {range_str}")
    print(f"ℹ  Existing rows found: {len(existing_rows)}")

    if added_binance or added_kraken:
        print(f" Raw data updated: {out_path}")
        if added_binance:
            print(f" Added {added_binance} candles from Binance")
        if added_kraken:
            print(f" Added {added_kraken} candles from Kraken")
        if kraken_limited:
            print(" Could not retrieve full range: Kraken limited to 720 candles")
    else:
        print(" Raw data already complete. No candles fetched.")


def fetch_missing_candles(tag: str, relative_window: str = "48h", verbose: int = 1) -> None:
    tag = tag.upper()

    try:
        symbols = resolve_symbol(tag)
        kraken_symbol = symbols["kraken"]
        binance_symbol = symbols["binance"]
    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed to resolve symbol '{tag}': {e}")

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

        if diff_hours <= 720:
            try:
                rows = _fetch_kraken(kraken_symbol, start_ms, end_ms)
                if rows:
                    df = pd.DataFrame(rows, columns=COLUMNS)
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).astype("int64") // 1_000_000_000
                    new_frames.append(df)
                    added_kraken += len(df)
            except Exception as e:
                if verbose >= 1:
                    tqdm.write(f"[WARN] Kraken fetch error: {e}")
        else:
            kraken_end_ms = start_ms + 720 * 3600000
            try:
                rows = _fetch_kraken(kraken_symbol, start_ms, kraken_end_ms)
                if rows:
                    df_k = pd.DataFrame(rows, columns=COLUMNS)
                    df_k["timestamp"] = pd.to_datetime(df_k["timestamp"], unit="ms", utc=True).astype("int64") // 1_000_000_000
                    new_frames.append(df_k)
                    added_kraken += len(df_k)
            except Exception as e:
                if verbose >= 1:
                    tqdm.write(f"[WARN] Kraken fetch error: {e}")

            try:
                rows = _fetch_binance(binance_symbol, kraken_end_ms + 3600000, end_ms)
                if rows:
                    df_b = pd.DataFrame(rows, columns=COLUMNS)
                    df_b["timestamp"] = pd.to_datetime(df_b["timestamp"], unit="ms", utc=True).astype("int64") // 1_000_000_000
                    new_frames.append(df_b)
                    added_binance += len(df_b)
            except Exception as e:
                if verbose >= 1:
                    tqdm.write(f"[WARN] Binance fetch error: {e}")

    fetch_and_store_combined(early_gap)
    fetch_and_store_combined(late_gap)

    _merge_and_save(out_path, existing, new_frames)

    if verbose >= 1:
        tqdm.write(f"[SYNC] Added {added_kraken} candles from Kraken, {added_binance} from Binance")


if __name__ == "__main__":
    main()
