from __future__ import annotations

import re
from datetime import timedelta, datetime, timezone
import os

import pandas as pd

from .metabrain.engine_utils import cache_all_brains, extract_features_at_t
from .metabrain.arbiter import run_arbiter

_INTERVAL_RE = re.compile(r'[_\-]((\d+)([smhdw]))(?=\.|_|$)', re.I)

TIMEFRAME_SECONDS = {
    's': 1,
    'm': 30 * 24 * 3600,  # month (≈30 days)
    'h': 3600,
    'd': 86400,
    'w': 604800,
}

INTERVAL_SECONDS = {
    's': 1,
    'm': 60,
    'h': 3600,
    'd': 86400,
    'w': 604800,
}

WINDOW_SIZE = 24
WINDOW_STEP = 2
CLUSTER_WINDOW = 10
BASE_SIZE = 10
SCALE_POWER = 2

def parse_timeframe(tf: str) -> timedelta | None:
    """Parse strings like '12h', '3d', '1m', '6w' into timedelta."""
    if not tf:
        return None
    m = re.match(r'(?i)^\s*(\d+)\s*([smhdw])\s*$', tf)
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
    """Filter dataframe to the requested timeframe."""
    if delta is None:
        return df
    if 'timestamp' in df.columns:
        ts = df['timestamp']
        ts_max = float(ts.iloc[-1])
        is_ms = ts_max > 1e12
        to_seconds = (ts / 1000.0) if is_ms else ts
        cutoff = (datetime.now(timezone.utc).timestamp() - delta.total_seconds())
        mask = to_seconds >= cutoff
        return df.loc[mask]
    for col in ('datetime','date','time'):
        if col in df.columns:
            try:
                dt = pd.to_datetime(df[col], utc=True, errors='coerce')
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


def multi_window_vote(df, t, window_sizes, slope_thresh=0.001, range_thresh=0.05):
    """Return (-1,0,1) decision with confidence using multi-window slope direction."""
    import numpy as np

    votes, strengths = [], []
    for W in window_sizes:
        if t - W < 0:
            continue
        sub = df.iloc[t - W : t]
        closes = sub["close"].values
        x = np.arange(len(closes))
        slope = float(np.polyfit(x, closes, 1)[0]) if len(closes) > 1 else 0.0
        rng = float(sub["close"].max() - sub["close"].min())
        if abs(slope) < slope_thresh or rng < range_thresh:
            continue
        direction = 1 if slope > 0 else -1
        votes.append(direction)
        strengths.append(abs(slope) * rng)
    score = sum(votes)
    confidence = (sum(strengths) / max(1, len(strengths))) if strengths else 0.0
    if score >= 2:
        return 1, confidence, score
    if score <= -2:
        return -1, confidence, score
    return 0, confidence, score

def run_simulation(timeframe: str = "1m", viz: bool = True) -> None:
    file_path = "data/sim/SOLUSD_1h.csv"
    df = pd.read_csv(file_path)
    delta = parse_timeframe(timeframe)
    if delta is not None:
        df = apply_time_filter(df, delta, file_path)
    df = df.reset_index(drop=True)
    df["candle_index"] = range(len(df))

    brain_cache = cache_all_brains(df)
    state = "flat"
    buy_signals, sell_signals = [], []
    decisions = []
    hold_counter = 0

    for t in range(50, len(df)):
        features = extract_features_at_t(brain_cache, t)
        decision, reasons = run_arbiter(features, state, debug=True)

        x = int(df["candle_index"].iloc[t])
        y = float(df["close"].iloc[t])
        include = True
        if state == "flat" and decision == "BUY":
            buy_signals.append((x, y))
            state = "long"
        elif state == "long" and decision == "SELL":
            sell_signals.append((x, y))
            state = "flat"
        else:
            if decision == "HOLD":
                hold_counter += 1
                if hold_counter % 10 != 0:
                    include = False

        if include:
            decisions.append((x, y, decision, reasons, features))

    if viz:
        import matplotlib.pyplot as plt
        import mplcursors

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df["candle_index"], df["close"], lw=1, color="blue")

        scatters = []
        for x, y, decision, _, _ in decisions:
            if decision == "BUY":
                sc = ax.scatter(x, y, color="green", marker="^", s=120, zorder=6)
            elif decision == "SELL":
                sc = ax.scatter(x, y, color="red", marker="v", s=120, zorder=6)
            else:
                sc = ax.scatter(
                    x, y, color="gray", marker="o", s=40, alpha=0.3, zorder=4
                )
            scatters.append(sc)

        cursor = mplcursors.cursor(scatters, hover=True)

        @cursor.connect("add")
        def on_hover(sel):
            idx = scatters.index(sel.artist)
            x, y, decision, reasons, feats = decisions[idx]
            lines = [f"{decision} @ {x}", f"Price={y:.2f}"]
            lines.extend(reasons)
            lines.append("features:")
            for k, v in sorted(feats.items()):
                lines.append(f"{k}={v}")
            sel.annotation.set_text("\n".join(lines))

        plt.show()

    print(f"[SIM][{timeframe}] buys={len(buy_signals)} sells={len(sell_signals)}")
