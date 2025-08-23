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

def load_candles(symbol: str) -> tuple[pd.DataFrame, str]:
    """Load full-resolution candles for ``symbol`` with 1h fallback."""
    base = Path("data/sim")
    csv = base / f"{symbol}.csv"
    if not csv.exists():
        csv = base / f"{symbol}_1h.csv"
    df = pd.read_csv(csv)
    return df, str(csv)


def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add common vectorized features to ``df``."""
    df = df.copy()
    df["return"] = df["close"].pct_change()
    df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
    delta = df["close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / roll_down
    df["rsi"] = 100 - (100 / (1 + rs))
    df["volatility"] = df["return"].rolling(20).std()
    rolling_mean = df["close"].rolling(20).mean()
    rolling_std = df["close"].rolling(20).std()
    df["zscore"] = (df["close"] - rolling_mean) / rolling_std
    high_col = "high" if "high" in df.columns else "close"
    low_col = "low" if "low" in df.columns else "close"
    df["rolling_high"] = df[high_col].rolling(20).max()
    df["rolling_low"] = df[low_col].rolling(20).min()
    return df


# -----------------------------------------------------------------------------
# Teaching engine
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


def run_teach(brain: str, timeframe: str, mode: str = "audit") -> None:
    """Run teaching/audit loop for a given brain."""
    df, file_path = load_candles("SOLUSD")
    delta = parse_timeframe(timeframe)
    if delta is not None:
        df = apply_time_filter(df, delta, file_path)
    df = df.reset_index(drop=True)
    df["candle_index"] = range(len(df))
    df = enrich_features(df)

    mod = importlib.import_module(f"systems.brains.{brain}")
    signals = mod.run(df, viz=False)

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
        feat["price"] = float(df["close"].iloc[t])
        rows.append(feat)

    counts = {"BUY": 0, "SELL": 0, "HOLD": 0}

    if mode == "audit":
        feat_df = pd.DataFrame(rows)
        if not feat_df.empty:
            print(feat_df.describe())
            print("Correlations:")
            print(feat_df.corr(numeric_only=True))
        counts["HOLD"] = len(rows)

    elif mode == "teach":
        import matplotlib.pyplot as plt

        labels_path = Path(f"data/labels/{brain}.jsonl")
        labels_path.parent.mkdir(parents=True, exist_ok=True)

        indices = [r["index"] for r in rows]
        prices = [r["price"] for r in rows]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df["candle_index"], df["close"], lw=1, color="blue")
        ax.scatter(indices, prices, color="orange", marker="o")

        state = {"i": 0}
        highlight = ax.scatter([], [], s=200, facecolors="none", edgecolors="green")

        if indices:
            highlight.set_offsets([[indices[0], prices[0]]])

        def on_key(event):
            if not indices:
                return
            i = state["i"]
            key = event.key.lower()
            if key in ("b", "s", "h"):
                label = {"b": "BUY", "s": "SELL", "h": "HOLD"}[key]
                out = rows[i].copy()
                out["label"] = label
                with labels_path.open("a") as fh:
                    fh.write(json.dumps(out) + "\n")
                counts[label] += 1
                state["i"] = min(len(indices) - 1, i + 1)
                new_i = state["i"]
                highlight.set_offsets([[indices[new_i], prices[new_i]]])
                fig.canvas.draw_idle()

        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.show()

    elif mode == "correct":
        labels_path = Path(f"data/labels/{brain}.jsonl")
        if labels_path.exists():
            lines = labels_path.read_text().splitlines()
            print(f"Loaded {len(lines)} existing labels for correction (not implemented)")
        else:
            print("No existing labels to correct.")

    total = sum(counts.values())
    print(
        f"[TEACH][{brain}][{timeframe}] "
        f"count={total} buy={counts['BUY']} sell={counts['SELL']} hold={counts['HOLD']}"
    )

