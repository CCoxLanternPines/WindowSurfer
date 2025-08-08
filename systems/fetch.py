from __future__ import annotations

"""Time-aware historical data fetcher."""

from datetime import datetime, timezone
from pathlib import Path
from typing import List
import sys
import pandas as pd
from systems.utils.config import load_ledger_config, load_settings, resolve_path
from systems.utils.cli import build_parser
from systems.utils.symbols import (
    resolve_asset,
    resolve_tag,
    raw_path,
)
from systems.utils.resolve_symbol import resolve_ledger_settings

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from systems.utils.time import parse_relative_time

from tqdm import tqdm
from systems.utils.addlog import addlog
from systems.scripts.fetch_core import (
    _load_existing,
    _merge_and_save,
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
    parser.add_argument("--tag", help="Trading pair tag", required=False)
    args = parser.parse_args(argv)
    if not args.ledger and not args.tag:
        parser.error("--ledger or --tag is required")

    time_window = args.time if args.time else "48h"
    verbose = args.verbose

    if args.tag:
        tag = args.tag.upper()
        ledger_cfg = resolve_ledger_settings(tag)
        addlog(
            f"[INFO] CLI tag provided; resolved ledger {ledger_cfg.get('asset', tag)}",
            verbose_int=1,
            verbose_state=True,
        )
    else:
        settings = load_settings()
        ledger_cfg = settings["ledger_settings"][args.ledger]
        tag = ledger_cfg.get("tag", args.ledger).upper()

    asset = resolve_asset(ledger_cfg)
    kraken_symbol = ledger_cfg.get("kraken_name")
    if not kraken_symbol:
        addlog(
            f"[WARN] Missing kraken_name for {asset} ({tag})",
            verbose_int=1,
            verbose_state=True,
        )
    binance_symbol = ledger_cfg.get("binance_name")
    if not binance_symbol:
        addlog(
            f"[WARN] Missing binance_name for {asset} ({tag})",
            verbose_int=1,
            verbose_state=True,
        )

    start_ts, end_ts = parse_relative_time(time_window)
    out_path = raw_path(asset)
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
            if kraken_symbol:
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
            if kraken_symbol:
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
            if binance_symbol:
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


def fetch_missing_candles(
    ledger: str,
    *,
    kraken_pair: str | None = None,
    relative_window: str = "48h",
    verbose: int = 1,
) -> None:
    ledger_cfg = load_ledger_config(ledger)
    tag = resolve_tag(ledger_cfg)
    asset = resolve_asset(ledger_cfg)
    kraken_symbol = kraken_pair or ledger_cfg.get("kraken_name")
    if not kraken_symbol:
        addlog(
            f"[WARN] Missing kraken_name for {asset} ({tag})",
            verbose_int=1,
            verbose_state=True,
        )
    binance_symbol = ledger_cfg.get("binance_name")
    if not binance_symbol:
        addlog(
            f"[WARN] Missing binance_name for {asset} ({tag})",
            verbose_int=1,
            verbose_state=True,
        )

    start_ts, end_ts = parse_relative_time(relative_window)
    out_path = raw_path(asset)
    existing = _load_existing(out_path)
    gaps = compute_missing_ranges(existing, start_ts, end_ts, 3_600_000)

    new_frames: List[pd.DataFrame] = []
    added_binance = 0
    added_kraken = 0

    for gap_start, gap_end in gaps:
        diff_hours = int((gap_end - gap_start) // 3600) + 1
        if diff_hours <= 720:
            if kraken_symbol:
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
            if kraken_symbol:
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
            if binance_symbol:
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
