from __future__ import annotations

"""Legacy utilities for refreshing candles (live mode deprecated)."""

from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pandas as pd

from systems.scripts.fetch_candles import fetch_candles, COLUMNS
from systems.utils.config import resolve_path

try:  # pragma: no cover - optional dependency
    from systems.utils.resolve_symbol import resolve_ccxt_symbols
except Exception:  # pragma: no cover - fallback
    def resolve_ccxt_symbols(settings, tag):  # type: ignore
        return tag, None

from systems.utils.addlog import addlog


def get_raw_path(tag: str, ext: str = "csv") -> Path:
    """Return the full path to the raw-data file for ``tag``."""

    root = resolve_path("")
    return root / "data" / "raw" / f"{tag}.{ext}"


def _load_existing(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        return df
    return pd.DataFrame(columns=COLUMNS)


def _merge_and_save(path: Path, existing: pd.DataFrame, new_frames: List[pd.DataFrame]) -> int:
    combined = pd.concat([existing] + new_frames, ignore_index=True)
    combined = combined.drop_duplicates(subset="timestamp").sort_values("timestamp")

    ts = combined["timestamp"].to_numpy()
    if len(ts) > 1:
        gaps = (ts[1:] - ts[:-1]) != 3600
        if gaps.any():
            missing_spans = int(gaps.sum())
            addlog(
                f"[WARN] Post-merge gaps detected: {missing_spans} hour(s) missing",
                verbose_state=True,
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    return len(combined)


def compute_missing_ranges(
    candles_df: pd.DataFrame, start_ts: int, end_ts: int, interval_ms: int
) -> List[tuple[int, int]]:
    """Return a list of (start, end) tuples for missing candle ranges."""

    interval_s = interval_ms // 1000
    if candles_df.empty:
        return [(start_ts, end_ts)]

    windowed = candles_df[
        (candles_df["timestamp"] >= start_ts)
        & (candles_df["timestamp"] <= end_ts)
    ]["timestamp"].sort_values()

    ranges: List[tuple[int, int]] = []
    current = start_ts
    for ts in windowed:
        if ts > current:
            ranges.append((current, min(ts - interval_s, end_ts)))
        current = ts + interval_s
        if current > end_ts:
            break

    if current <= end_ts:
        ranges.append((current, end_ts))

    return ranges


def refresh_to_last_closed_hour(
    settings,
    tag: str,
    *,
    exchange: str = "kraken",
    lookback_hours: int = 72,
    verbose: int = 1,
) -> None:
    """Deprecated helper retained for legacy scripts."""

    raise RuntimeError(
        "refresh_to_last_closed_hour is deprecated; use refresh_live_kraken_720 from candle_cache"
    )

