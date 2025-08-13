from __future__ import annotations

"""Utilities to refresh missing hourly candles before live runs."""

from datetime import datetime, timezone
import pandas as pd

from systems.scripts.fetch_core import (
    get_raw_path,
    _load_existing,
    _merge_and_save,
    compute_missing_ranges,
    fetch_range,
)

try:  # pragma: no cover - optional dependency
    from systems.utils.resolve_symbol import resolve_ccxt_symbols
except Exception:  # pragma: no cover - fallback
    def resolve_ccxt_symbols(settings, tag):  # type: ignore
        return tag, None

from systems.utils.addlog import addlog


def refresh_to_last_closed_hour(
    settings,
    tag: str,
    *,
    coin: str | None = None,
    exchange: str = "kraken",
    lookback_hours: int = 72,
    verbose: int = 1,
) -> None:
    """Ensure ``data/raw/{coin or tag}.csv`` has candles up to the last closed hour."""

    kraken_sym, _ = resolve_ccxt_symbols(settings, tag)

    path = get_raw_path((coin or tag), ext="csv")
    existing = _load_existing(path)

    now = datetime.now(timezone.utc)
    last_closed = int((now.timestamp() // 3600 - 1) * 3600)
    start_ts = last_closed - lookback_hours * 3600
    if not existing.empty:
        latest_ts = int(existing["timestamp"].max())
        start_ts = max(start_ts, latest_ts - 6 * 3600)

    missing = compute_missing_ranges(existing, start_ts, last_closed, 3600 * 1000)

    frames: list[pd.DataFrame] = []
    for s, e in missing:
        if e < s:
            continue
        df = fetch_range(exchange, kraken_sym, s, e)
        if not df.empty:
            frames.append(df)

    if frames:
        count = _merge_and_save(path, existing, frames)
        addlog(
            f"[FETCH][LIVE] {tag} merged → {count} rows",
            verbose_int=verbose,
            verbose_state=True,
        )
    else:
        addlog(
            f"[FETCH][LIVE] {tag} up-to-date (<= {last_closed})",
            verbose_int=verbose,
            verbose_state=True,
        )

