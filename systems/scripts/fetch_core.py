from __future__ import annotations

from pathlib import Path
from typing import List, Iterable, Tuple

import pandas as pd
import ccxt

from systems.utils.config import resolve_path, load_settings
from systems.utils.resolve_symbol import resolve_symbol

COLUMNS = ["ts", "open", "high", "low", "close", "volume"]
HOUR_MS = 3_600_000
LOOKBACK = 720


class FetchAbort(Exception):
    """Raised when candle data cannot be healed automatically."""


def _fetch_kraken(symbol: str, start_ms: int, end_ms: int) -> List[List]:
    exchange = ccxt.kraken({"enableRateLimit": True})
    limit = min(720, int((end_ms - start_ms) // HOUR_MS) + 1)
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe="1h", since=start_ms, limit=limit)
    return [row for row in ohlcv if row and start_ms <= row[0] <= end_ms]


def _fetch_binance(symbol: str, start_ms: int, end_ms: int) -> List[List]:
    exchange = ccxt.binance({"enableRateLimit": True})
    rows: List[List] = []
    since = start_ms
    limit = 1000
    while since <= end_ms:
        chunk = exchange.fetch_ohlcv(symbol, timeframe="1h", since=since, limit=limit)
        if not chunk:
            break
        for r in chunk:
            if r and since <= r[0] <= end_ms:
                rows.append(r)
        last = chunk[-1][0]
        if last >= end_ms:
            break
        since = last + HOUR_MS
    return rows


def _load_cache(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_parquet(path)
        return _canonicalize(df)
    return pd.DataFrame(columns=COLUMNS)


def _write_cache(path: Path, df: pd.DataFrame) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def _canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out.columns = [c.lower() for c in out.columns]
    if "timestamp" in out.columns and "ts" not in out.columns:
        out = out.rename(columns={"timestamp": "ts"})
    out["ts"] = (out["ts"].astype("int64") // HOUR_MS) * HOUR_MS
    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["volume"] = out["volume"].fillna(0.0)
    out = out.dropna(subset=COLUMNS)
    out = out.sort_values("ts").drop_duplicates("ts", keep="first")
    return out[COLUMNS]


def _find_gaps(df: pd.DataFrame, last_complete: int) -> List[Tuple[int, int, int]]:
    if df.empty:
        return []
    start = int(df["ts"].min())
    expected = range(start, last_complete + HOUR_MS, HOUR_MS)
    existing = set(int(ts) for ts in df["ts"])
    missing = [ts for ts in expected if ts not in existing]
    gaps: List[Tuple[int, int, int]] = []
    if not missing:
        return gaps
    s = missing[0]
    prev = missing[0]
    count = 1
    for ts in missing[1:]:
        if ts == prev + HOUR_MS:
            prev = ts
            count += 1
        else:
            gaps.append((s, prev, count))
            s = prev = ts
            count = 1
    gaps.append((s, prev, count))
    return gaps


def _log_cache(df: pd.DataFrame) -> None:
    if df.empty:
        print("[FETCH] cache rows=0 span=[none] gaps=0")
        return
    first = int(df["ts"].min())
    last = int(df["ts"].max())
    gaps = _find_gaps(df, last)
    fstr = pd.to_datetime(first, unit="ms", utc=True).strftime("%Y-%m-%d")
    lstr = pd.to_datetime(last, unit="ms", utc=True).strftime("%Y-%m-%d")
    print(f"[FETCH] cache rows={len(df)} span=[{fstr},{lstr}] gaps={len(gaps)}")


def _heal_gap(df: pd.DataFrame, gap: Tuple[int, int, int], kraken_symbol: str) -> Tuple[pd.DataFrame, int]:
    start, end, length = gap
    print(
        f"[HEAL] recent gap len={length} candles range=[{pd.to_datetime(start, unit='ms', utc=True)},"
        f"{pd.to_datetime(end, unit='ms', utc=True)}] via Kraken..."
    )
    rows = _fetch_kraken(kraken_symbol, start, end)
    heal_df = _canonicalize(pd.DataFrame(rows, columns=COLUMNS)) if rows else pd.DataFrame(columns=COLUMNS)
    if heal_df.empty:
        return df, 0
    merged = pd.concat([df, heal_df], ignore_index=True)
    merged = _canonicalize(merged)
    return merged, len(heal_df)


def get_gapless_1h(tag: str, allow_cache: bool = True) -> pd.DataFrame:
    """Return a gapless 1h DataFrame for ``tag``."""

    symbols = resolve_symbol(tag)
    binance_symbol = symbols["binance"]
    kraken_symbol = symbols["kraken"]

    path = resolve_path(f"data/raw/{tag}_1h.parquet")
    cache_df = _load_cache(path) if allow_cache else pd.DataFrame(columns=COLUMNS)
    _log_cache(cache_df)

    now_ms = int(pd.Timestamp.utcnow().floor("1h").timestamp() * 1000)
    last_complete = now_ms - HOUR_MS

    if cache_df.empty or not allow_cache:
        start_ms = 0
    else:
        start_ms = int(cache_df["ts"].max()) + HOUR_MS
    if start_ms <= last_complete:
        rows = _fetch_binance(binance_symbol, start_ms, last_complete)
        new_df = _canonicalize(pd.DataFrame(rows, columns=COLUMNS)) if rows else pd.DataFrame(columns=COLUMNS)
        cache_df = _canonicalize(pd.concat([cache_df, new_df], ignore_index=True))

    gaps = _find_gaps(cache_df, last_complete)
    recent_threshold = last_complete - LOOKBACK * HOUR_MS
    recent = [g for g in gaps if g[1] >= recent_threshold]
    older = [g for g in gaps if g[1] < recent_threshold]

    while recent:
        gap = recent[0]
        cache_df, filled = _heal_gap(cache_df, gap, kraken_symbol)
        gaps = _find_gaps(cache_df, last_complete)
        recent = [g for g in gaps if g[1] >= recent_threshold]
        older = [g for g in gaps if g[1] < recent_threshold]
        print(f"[HEAL] filled={filled} candles (remaining gaps={len(gaps)})")
        if filled == 0:
            break

    if older or recent:
        msg = (
            "[ABORT][FETCH] Gap(s) detected outside the last 720 candles or not healable.\n"
            "Action: run full refresh → python systems/fetch.py --tag {tag} --full\n"
            f"Details: gaps={gaps}"
        )
        raise FetchAbort(msg)

    start = int(cache_df["ts"].min())
    idx = range(start, last_complete + HOUR_MS, HOUR_MS)
    final_df = cache_df.set_index("ts").reindex(idx)
    if final_df.isnull().any().any():
        raise FetchAbort("[ABORT][FETCH] unresolved gaps after healing")
    final_df = final_df.reset_index().rename(columns={"index": "ts"})
    final_df = _canonicalize(final_df)
    _write_cache(path, final_df)

    fstr = pd.to_datetime(final_df["ts"].min(), unit="ms", utc=True).strftime("%Y-%m-%d")
    lstr = pd.to_datetime(final_df["ts"].max(), unit="ms", utc=True).strftime("%Y-%m-%d")
    print(f"[FETCH] gapless ✓ rows={len(final_df)} span=[{fstr},{lstr}]")

    return final_df

