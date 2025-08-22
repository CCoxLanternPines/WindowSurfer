from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .brains import load_brain

# ===================== Parameters =====================
WINDOW_STEP = 2
SLOPE_WIN = 12

# ===================== Helpers (copied from sim_engine) =====================
_INTERVAL_RE = re.compile(r'[_\-]((\d+)([smhdw]))(?=\.|_|$)', re.I)

UNIT_SECONDS = {
    's': 1,
    # interpret 'm' as months (~30 days)
    'm': 30 * 24 * 3600,
    'h': 3600,
    'd': 86400,
    'w': 604800,
}

def parse_timeframe(tf: str) -> timedelta | None:
    """Parse strings like '12h', '3d', '6w', '3m' into timedelta."""
    if not tf:
        return None
    m = re.match(r'(?i)^\s*(\d+)\s*([smhdw])\s*$', tf)
    if not m:
        return None
    n, u = int(m.group(1)), m.group(2).lower()
    return timedelta(seconds=n * UNIT_SECONDS[u])

def infer_candle_seconds_from_filename(path: str) -> int | None:
    """Try to infer candle interval from filename like *_1h.csv, *_15m.csv, *_1d.csv."""
    m = _INTERVAL_RE.search(os.path.basename(path))
    if not m:
        return None
    n, u = int(m.group(2)), m.group(3).lower()
    return n * UNIT_SECONDS[u]

def apply_time_filter(df: pd.DataFrame, delta: timedelta, file_path: str, warmup: int) -> pd.DataFrame:
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
    need = int(max(warmup + 1, delta.total_seconds() // sec))
    if need <= 0 or need >= len(df):
        return df
    return df.iloc[-need:]

# ===================== Trend Helper =====================
def multi_window_vote(df, t, window_sizes, slope_thresh=0.001, range_thresh=0.05):
    """Return (-1,0,1) decision with confidence using multi-window slope direction."""
    votes, strengths = [], []
    for W in window_sizes:
        if t - W < 0:
            continue
        sub = df.iloc[t - W:t]
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

# ===================== Runner =====================
def run(brain_name: str, timeframe: str, viz: bool = True) -> dict[str, float]:
    file_path = "data/sim/SOLUSD_1h.csv"
    df = pd.read_csv(file_path)

    brain = load_brain(brain_name)

    delta = parse_timeframe(timeframe)
    if delta is not None:
        df = apply_time_filter(df, delta, file_path, brain.warmup())
    df = df.reset_index(drop=True)
    df["candle_index"] = range(len(df))

    brain.prepare(df)

    trend_state: list[int] = []
    angles: list[float] = []
    for t in range(len(df)):
        decision, _, _ = multi_window_vote(df, t, window_sizes=[8, 12, 24, 48])
        if decision == 1:
            trend_state.append(1)
        elif decision == -1:
            trend_state.append(-1)
        else:
            trend_state.append(0)
        if t >= SLOPE_WIN:
            sub = df["close"].iloc[t - SLOPE_WIN + 1 : t + 1]
            slope = float(np.polyfit(np.arange(len(sub)), sub, 1)[0]) if len(sub) > 1 else 0.0
            angle = math.degrees(math.atan(slope))
        else:
            angle = 0.0
        angles.append(angle)

    for t in range(brain.warmup(), len(df), WINDOW_STEP):
        brain.step(df, t)

    pts = brain.overlays()
    if brain_name == "exhaustion":
        pts = {k: v for k, v in pts.items() if k.startswith("exhaustion")}

    stats = brain.compute_stats(df, trend_state, angles)

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("data/out") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "brain_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("Brain statistics:")
    for k, d in stats.items():
        print(f"{k:40s} [{d['count']}/{d['total']}] {d['value']:.4f}")

    if not viz:
        return stats

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df["candle_index"], df["close"], lw=1, label="Close Price", color="blue")

    styles = {
        "reversal": dict(c="yellow", s=120, edgecolors="black", zorder=7),
        "bottom4": dict(c="cyan", marker="v", s=100, zorder=6),
        "top5": dict(c="orange", marker="s", s=110, zorder=6),
        "top6": dict(c="red", marker="*", s=140, zorder=6),
        "top7": dict(c="purple", marker="^", s=110, zorder=6),
        "top8": dict(c="magenta", marker="P", s=160, zorder=7),
        "valley_w": dict(c="teal", marker="h", s=120, zorder=7),
        "valley_e": dict(c="deepskyblue", marker="D", s=110, zorder=7),
        "valley_r": dict(c="darkcyan", marker="s", s=100, zorder=7),
        "valley_t": dict(c="turquoise", marker="P", s=150, zorder=8),
    }

    for name, data in pts.items():
        x = data.get("x", [])
        y = data.get("y", [])
        if not x or not y:
            continue
        style = styles.get(name, {})
        s = data.get("s", style.get("s"))
        c = data.get("c", style.get("c"))
        marker = style.get("marker", "o")
        edgecolors = style.get("edgecolors")
        zorder = style.get("zorder", 6)
        ax1.scatter(x, y, s=s, c=c, marker=marker, edgecolors=edgecolors, zorder=zorder)

    ax1.set_title(f"Price with {brain_name.title()} overlays")
    ax1.set_xlabel("Candles (Index)")
    ax1.set_ylabel("Price")
    ax1.grid(True)

    plt.show()

    return stats
