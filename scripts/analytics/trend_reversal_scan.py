#!/usr/bin/env python3
"""Trend reversal scanner for candle CSVs.

This standalone utility scans candle data to identify streaks of
window-position extremes and checks for immediate reversals.

Examples
--------
# 3d window (72 bars if 1h), streaks ≥ 3, save details
python scripts/analytics/trend_reversal_scan.py \
  --csv data/raw/SOLUSDC_1h.csv \
  --window 3d \
  --streak 3 \
  --save data/tmp/solusdc_streaks.csv

# Custom thresholds, integer window size
python scripts/analytics/trend_reversal_scan.py \
  --csv data/raw/SOLUSDT_1h.csv \
  --window 96 \
  --streak 4 \
  --pos-low 0.25 \
  --pos-high 0.75
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

WINDOW_ALIASES = {
    "1d": 24,
    "2d": 48,
    "3d": 72,
    "6d": 144,
    "1w": 168,
}


def parse_window(value: str) -> int:
    """Return integer window size in bars for ``value``."""
    if isinstance(value, int):
        return value
    val = value.strip().lower()
    if val in WINDOW_ALIASES:
        return WINDOW_ALIASES[val]
    try:
        return int(val)
    except ValueError as exc:  # pragma: no cover - user input
        raise argparse.ArgumentTypeError(f"invalid window value: {value}") from exc


def resolve_time_column(df: pd.DataFrame, name: Optional[str]) -> Optional[str]:
    """Return the column name to use for timestamps, if available."""
    if name:
        lower = [c.lower() for c in df.columns]
        for orig, low in zip(df.columns, lower):
            if low == name.lower():
                return orig
        raise ValueError(f"time column '{name}' not found")
    for candidate in ("timestamp", "time"):
        for orig in df.columns:
            if orig.lower() == candidate:
                return orig
    return None


def compute_positions(close: pd.Series, window: int) -> pd.Series:
    """Compute WindowSurfer-style rolling window positions."""
    roll_min = close.rolling(window, min_periods=window).min()
    roll_max = close.rolling(window, min_periods=window).max()
    rng = roll_max - roll_min
    pos = (close - roll_min) / rng
    pos[rng == 0] = np.nan
    return pos.clip(0.0, 1.0)


def find_streaks(positions: Iterable[float], pos_low: float, pos_high: float, min_len: int) -> List[Tuple[str, int, int]]:
    """Return streak records as ``(kind, start_idx, end_idx)``."""
    records: List[Tuple[str, int, int]] = []
    kind: Optional[str] = None
    start: Optional[int] = None
    for i, p in enumerate(positions):
        if np.isnan(p):
            if kind is not None and i - start >= min_len:
                records.append((kind, start, i - 1))
            kind = None
            start = None
            continue
        if p < pos_low:
            if kind == "down":
                continue
            if kind is not None and i - start >= min_len:
                records.append((kind, start, i - 1))
            kind = "down"
            start = i
        elif p > pos_high:
            if kind == "up":
                continue
            if kind is not None and i - start >= min_len:
                records.append((kind, start, i - 1))
            kind = "up"
            start = i
        else:
            if kind is not None and i - start >= min_len:
                records.append((kind, start, i - 1))
            kind = None
            start = None
    if kind is not None and len(positions) - start >= min_len:
        records.append((kind, start, len(positions) - 1))
    return records


def evaluate_streaks(records: List[Tuple[str, int, int]], positions: pd.Series, pos_low: float, pos_high: float) -> List[dict]:
    """Return streak dicts with evaluation results."""
    out: List[dict] = []
    for kind, start, end in records:
        if end + 2 >= len(positions):
            result = "undetermined"
        else:
            next_two = positions.iloc[end + 1 : end + 3]
            if kind == "down":
                result = "win" if np.all(next_two > pos_high) else "loss"
            else:
                result = "win" if np.all(next_two < pos_low) else "loss"
        out.append({
            "kind": kind,
            "start_idx": start,
            "end_idx": end,
            "length": end - start + 1,
            "success": result,
        })
    return out


def print_summary(records: List[dict]) -> None:
    for label in ("down", "up"):
        subset = [r for r in records if r["kind"] == label]
        total = len(subset)
        wins = sum(1 for r in subset if r["success"] == "win")
        losses = sum(1 for r in subset if r["success"] == "loss")
        und = sum(1 for r in subset if r["success"] == "undetermined")
        win_pct = (wins / total * 100.0) if total else 0.0
        loss_pct = (losses / total * 100.0) if total else 0.0
        title = "Downtrend" if label == "down" else "Uptrend"
        print(f"=== {title} streaks ===")
        print(
            f"Total: {total} | Wins: {wins} ({win_pct:.1f}%) | Losses: {losses} ({loss_pct:.1f}%) | Undetermined: {und}"
        )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Scan candle CSV for trend streaks and reversals")
    parser.add_argument("--csv", required=True, help="path to candle CSV")
    parser.add_argument("--window", default="3d", help="rolling window size or alias")
    parser.add_argument("--streak", type=int, default=3, help="minimum streak length")
    parser.add_argument("--pos-low", type=float, default=0.3, dest="pos_low", help="low position threshold")
    parser.add_argument("--pos-high", type=float, default=0.7, dest="pos_high", help="high position threshold")
    parser.add_argument("--time-col", help="optional timestamp column name")
    parser.add_argument("--save", help="path to save detailed streak CSV")

    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        parser.print_help()
        print(
            "\nExample:\n  python scripts/analytics/trend_reversal_scan.py --csv data/raw/SOLUSDC.csv --window 3d --streak 3"
        )
        return 0
    args = parser.parse_args(argv)

    window = parse_window(args.window)
    df = pd.read_csv(args.csv)
    df.columns = [c.lower() for c in df.columns]
    if "close" not in df.columns:
        raise ValueError("CSV must contain a 'close' column")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)

    time_col_name = resolve_time_column(df, args.time_col)
    if time_col_name:
        time_series = df[time_col_name]
    else:
        time_series = None

    pos = compute_positions(df["close"], window)
    streaks = find_streaks(pos, args.pos_low, args.pos_high, args.streak)
    evaluated = evaluate_streaks(streaks, pos, args.pos_low, args.pos_high)

    if time_series is not None:
        for rec in evaluated:
            rec["start_time"] = time_series.iloc[rec["start_idx"]]
            rec["end_time"] = time_series.iloc[rec["end_idx"]]

    print_summary(evaluated)

    if args.save and evaluated:
        save_path = os.path.abspath(args.save)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        pd.DataFrame(evaluated).to_csv(save_path, index=False)
        print(f"Saved details to {save_path}")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
