from __future__ import annotations

"""Core helpers for deterministic, gapless historical fetches."""

from pathlib import Path
from typing import Iterable, List, Tuple

import ccxt
import pandas as pd

from systems.utils.config import resolve_path

COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


class FetchAbort(RuntimeError):
    """Raised when a deterministic fetch cannot be completed."""


# ---------------------------------------------------------------------------
# fetching
# ---------------------------------------------------------------------------

def _fetch_kraken(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    exchange = ccxt.kraken({"enableRateLimit": True})
    limit = min(720, int((end_ms - start_ms) // 3_600_000))
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe="1h", since=start_ms, limit=limit)
    rows = [row for row in ohlcv if row and start_ms <= row[0] < end_ms]
    df = pd.DataFrame(rows, columns=COLUMNS)
    df = canonicalize(df)
    if df.empty:
        raise FetchAbort(
            f"0 candles returned symbol={symbol} span=[{start_ms},{end_ms}]"
        )
    return df


def _fetch_binance(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    exchange = ccxt.binance({"enableRateLimit": True})
    limit = 1_000
    rows: List[List] = []
    current = start_ms
    while current < end_ms:
        chunk = exchange.fetch_ohlcv(symbol, timeframe="1h", since=current, limit=limit)
        if not chunk:
            break
        filtered = [r for r in chunk if r and current <= r[0] < end_ms]
        if not filtered:
            break
        rows.extend(filtered)
        last = filtered[-1][0]
        current = last + 3_600_000
    df = pd.DataFrame(rows, columns=COLUMNS)
    df = canonicalize(df)
    if df.empty:
        try:
            recent = exchange.fetch_ohlcv(symbol, timeframe="1h", limit=1)
            if recent:
                first_ts = recent[0][0]
                if first_ts > start_ms:
                    pass  # listing is newer than requested span
        except Exception:
            pass
        if symbol.upper().endswith("USDC"):
            print(
                f"[HINT] Binance {symbol} may have limited history. "
                "Try SOLUSDT or reduce --time ≤ 720 to use Kraken."
            )
        raise FetchAbort(
            f"0 candles returned symbol={symbol} span=[{start_ms},{end_ms}]"
        )
    return df


# ---------------------------------------------------------------------------
# dataframe helpers
# ---------------------------------------------------------------------------

def canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure canonical columns and hourly UTC timestamps in ms."""
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    df = df.copy()
    df.columns = COLUMNS
    df["timestamp"] = (df["timestamp"].astype("int64") // 3_600_000) * 3_600_000
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
    return df.reset_index(drop=True)


def _find_gaps(ts: Iterable[int]) -> List[Tuple[int, int, int]]:
    ts_list = sorted(ts)
    gaps: List[Tuple[int, int, int]] = []
    for prev, nxt in zip(ts_list, ts_list[1:]):
        if nxt - prev > 3_600_000:
            start = prev + 3_600_000
            end = nxt - 3_600_000
            k = int((end - start) // 3_600_000 + 1)
            gaps.append((start, end, k))
    return gaps


def reindex_hourly(
    df: pd.DataFrame, start_ms: int, end_ms: int
) -> Tuple[pd.DataFrame, List[Tuple[int, int, int]]]:
    """Reindex ``df`` to an hourly grid and return gaps."""
    idx = range(start_ms, end_ms, 3_600_000)
    if df.empty:
        full = pd.DataFrame(index=idx, columns=COLUMNS)
        full.index.name = "timestamp"
        full = full.reset_index()
        gap_end = end_ms - 3_600_000 if end_ms > start_ms else start_ms
        gaps = [(start_ms, gap_end, len(idx))] if len(idx) else []
        return full, gaps

    df = df.set_index("timestamp").reindex(idx)
    df.index.name = "timestamp"
    full = df.reset_index()
    gaps = []
    i = 0
    while i < len(full):
        if pd.isna(full.loc[i, "open"]):
            gap_start = full.loc[i, "timestamp"]
            while i < len(full) and pd.isna(full.loc[i, "open"]):
                i += 1
            gap_end = full.loc[i - 1, "timestamp"]
            k = int((gap_end - gap_start) // 3_600_000 + 1)
            gaps.append((gap_start, gap_end, k))
        else:
            i += 1
    return full, gaps


# ---------------------------------------------------------------------------
# path helper
# ---------------------------------------------------------------------------

def get_raw_path(tag: str) -> Path:
    root = resolve_path("")
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir / f"{tag}_1h.parquet"


# ---------------------------------------------------------------------------
# public entry
# ---------------------------------------------------------------------------

def get_gapless_1h_for_span(
    ledger_cfg: dict, start_ms: int, end_ms: int
) -> pd.DataFrame:
    """Return a gapless DataFrame for the requested span applying policy A/B."""

    binance_symbol = ledger_cfg["binance_name"]
    kraken_symbol = ledger_cfg["kraken_name"]
    candles = int((end_ms - start_ms) // 3_600_000)

    if candles <= 720:
        df_k = _fetch_kraken(kraken_symbol, start_ms, end_ms)
        df_full, gaps = reindex_hourly(df_k, start_ms, end_ms)
        if gaps:
            df_b = _fetch_binance(binance_symbol, start_ms, end_ms)
            df_merge = canonicalize(pd.concat([df_k, df_b], ignore_index=True))
            df_full, gaps = reindex_hourly(df_merge, start_ms, end_ms)
            if gaps:
                raise FetchAbort(
                    "[ABORT][FETCH] Unhealable gaps in ≤720 span (Kraken primary). "
                    f"details={gaps}"
                )
        return df_full

    # Policy B: >720 candles
    df_b = _fetch_binance(binance_symbol, start_ms, end_ms)
    df_full, gaps = reindex_hourly(df_b, start_ms, end_ms)
    if gaps:
        recent_threshold = end_ms - 720 * 3_600_000
        recent = [g for g in gaps if g[1] >= recent_threshold]
        older = [g for g in gaps if g[1] < recent_threshold]
        if older:
            raise FetchAbort(
                "[ABORT][FETCH] Old gaps outside the last 720 candles. Run a full refresh or shorten --time."
                f" details={older}"
            )
        df_current = df_b
        for gstart, gend, _ in recent:
            df_k = _fetch_kraken(kraken_symbol, gstart, gend)
            df_current = canonicalize(pd.concat([df_current, df_k], ignore_index=True))
        df_full, gaps = reindex_hourly(df_current, start_ms, end_ms)
        if gaps:
            raise FetchAbort(
                "[ABORT][FETCH] Unhealable gaps after auto-heal. details="
                f"{gaps}"
            )
    return df_full
