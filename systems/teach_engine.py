from __future__ import annotations

"""Engine for auditing and teaching individual brains."""

import importlib
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd

# -----------------------------------------------------------------------------
# Timeframe helpers (cloned from systems.sim_engine)
# -----------------------------------------------------------------------------

_INTERVAL_RE = re.compile(r"[_\-]((\d+)([smhdw]))(?=\.|_|$)", re.I)

TIMEFRAME_SECONDS = {
    "s": 1,
    "m": 30 * 24 * 3600,  # month (≈30 days)
    "h": 3600,
    "d": 86400,
    "w": 604800,
}

INTERVAL_SECONDS = {
    "s": 1,
    "m": 60,
    "h": 3600,
    "d": 86400,
    "w": 604800,
}


def parse_timeframe(tf: str) -> timedelta | None:
    """Parse strings like '12h', '3d', '1m', '6w' into ``timedelta``."""
    if not tf:
        return None
    m = re.match(r"(?i)^\s*(\d+)\s*([smhdw])\s*$", tf)
    if not m:
        return None
    n, u = int(m.group(1)), m.group(2).lower()
    return timedelta(seconds=n * TIMEFRAME_SECONDS[u])


def infer_candle_seconds_from_filename(path: str) -> int | None:
    m = _INTERVAL_RE.search(os.path.basename(path))
    if not m:
        return None
    n, u = int(m.group(2)), m.group(3).lower()
    return n * INTERVAL_SECONDS[u]


def apply_time_filter(df: pd.DataFrame, delta: timedelta, file_path: str) -> pd.DataFrame:
    """Filter ``df`` to the most recent ``delta`` worth of candles."""
    if delta is None:
        return df
    if "timestamp" in df.columns:
        ts = df["timestamp"]
        ts_max = float(ts.iloc[-1])
        is_ms = ts_max > 1e12
        to_seconds = (ts / 1000.0) if is_ms else ts
        cutoff = datetime.now(timezone.utc).timestamp() - delta.total_seconds()
        mask = to_seconds >= cutoff
        return df.loc[mask]
    for col in ("datetime", "date", "time"):
        if col in df.columns:
            try:
                dt = pd.to_datetime(df[col], utc=True, errors="coerce")
                cutoff_dt = pd.Timestamp.utcnow() - delta
                mask = dt >= cutoff_dt
                return df.loc[mask]
            except Exception:
                pass
    sec = infer_candle_seconds_from_filename(file_path) or 3600
    need = int(delta.total_seconds() // sec)
    if need <= 0 or need >= len(df):
        return df
    return df.iloc[-need:]


# -----------------------------------------------------------------------------
# Data loading and feature engineering
# -----------------------------------------------------------------------------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Force lowercase column names for consistency."""
    return df.rename(columns={c: c.lower() for c in df.columns})


def load_candles(symbol: str) -> tuple[pd.DataFrame, str]:
    """Load full-resolution candles for ``symbol`` with 1h fallback."""
    base = Path("data/sim")
    csv = base / f"{symbol}.csv"
    if not csv.exists():
        csv = base / f"{symbol}_1h.csv"
    df = pd.read_csv(csv)
    df = normalize_columns(df)
    return df, str(csv)


def pick_price_column(df: pd.DataFrame, candidates: list[str], fallback: str | None = None) -> str:
    """Find a usable price column among candidates, else fallback."""
    for c in candidates:
        if c in df.columns:
            return c
    if fallback and fallback in df.columns:
        return fallback
    # last resort: first numeric column
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            print(f"[WARN] Falling back to column '{c}' as price source")
            return c
    raise ValueError("No usable price column found in candles DataFrame")


def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add common vectorized features to ``df``."""
    df = df.copy()
    close_col = pick_price_column(df, ["close", "closing_price", "c"])
    high_col = pick_price_column(df, ["high", "h"], close_col)
    low_col = pick_price_column(df, ["low", "l"], close_col)

    df["return"] = df[close_col].pct_change()
    df["ema_12"] = df[close_col].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df[close_col].ewm(span=26, adjust=False).mean()
    delta = df[close_col].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / roll_down
    df["rsi"] = 100 - (100 / (1 + rs))
    df["volatility"] = df["return"].rolling(20).std()
    rolling_mean = df[close_col].rolling(20).mean()
    rolling_std = df[close_col].rolling(20).std()
    df["zscore"] = (df[close_col] - rolling_mean) / rolling_std
    df["rolling_high"] = df[high_col].rolling(20).max()
    df["rolling_low"] = df[low_col].rolling(20).min()
    # unify reference for downstream
    df["price"] = df[close_col]
    return df


# -----------------------------------------------------------------------------
# Teaching engine core
# -----------------------------------------------------------------------------

FEATURE_COLS = [
    "return",
    "ema_12",
    "ema_26",
    "rsi",
    "volatility",
    "zscore",
    "rolling_high",
    "rolling_low",
]

def _run(brain: str, timeframe: str, viz: bool, mode: str) -> None:
    df, file_path = load_candles("SOLUSD")
    delta = parse_timeframe(timeframe)
    if delta is not None:
        df = apply_time_filter(df, delta, file_path)
    df = df.reset_index(drop=True)
    df["candle_index"] = range(len(df))
    df = enrich_features(df)

    mod = importlib.import_module(f"systems.brains.{brain}")
    signals = mod.run(df, viz=False)

    print(f"[DEBUG] Brain '{brain}' produced {len(signals)} signals")

    idx_map: Dict[int, Dict[str, float]] = {}
    for s in signals:
        idx = s.get("index") or s.get("candle_index")
        if idx is not None:
            idx_map[int(idx)] = s

    rows: List[Dict[str, float]] = []
    for t in range(len(df)):
        if t not in idx_map:
            continue
        feat = df.loc[t, FEATURE_COLS].to_dict()
        feat["index"] = int(df["candle_index"].iloc[t])
        feat["price"] = float(df["price"].iloc[t])
        rows.append(feat)

    counts = {"BUY": 0, "SELL": 0, "HOLD": 0}

    if mode == "audit":
        if not rows:
            print(f"[WARN] No matching candles found for brain '{brain}'")
        else:
            feat_df = pd.DataFrame(rows)
            print(feat_df.describe())
            print("Correlations:")
            print(feat_df.corr(numeric_only=True))
            counts["HOLD"] = len(rows)


# -----------------------------------------------------------------------------
# Public entrypoints
# -----------------------------------------------------------------------------

def run_audit(brain: str, timeframe: str):
    _run(brain, timeframe, viz=False, mode="audit")

def run_teach(brain: str, timeframe: str, viz: bool):
    _run(brain, timeframe, viz=viz, mode="teach")

def run_correct(brain: str, timeframe: str, viz: bool):
    _run(brain, timeframe, viz=viz, mode="correct")
