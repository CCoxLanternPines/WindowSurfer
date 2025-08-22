from __future__ import annotations

import argparse
import re
from datetime import timedelta, datetime, timezone
import os
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


sys.path.append(str(Path(__file__).resolve().parent))
from brains import load_brain

# ===================== Parameters =====================
WINDOW_STEP = 2  # step between boxes

# ===================== Helpers =====================
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
    """Robust timeframe filtering:
       1) If 'timestamp' column exists (seconds or ms), filter by UTC now - delta.
       2) Else if parseable datetime column exists (time/date/datetime), filter by that.
       3) Else fall back to row-count slicing using inferred candle interval from filename
          (e.g., *_1h.csv)."""
    if delta is None:
        return df

    # 1) Epoch timestamp (seconds or ms)
    if 'timestamp' in df.columns:
        ts = df['timestamp']
        # detect ms vs s
        ts_max = float(ts.iloc[-1])
        is_ms = ts_max > 1e12
        to_seconds = (ts / 1000.0) if is_ms else ts
        cutoff = (datetime.now(timezone.utc).timestamp() - delta.total_seconds())
        mask = to_seconds >= cutoff
        return df.loc[mask]

    # 2) Datetime-like columns
    for col in ('datetime','date','time'):
        if col in df.columns:
            try:
                dt = pd.to_datetime(df[col], utc=True, errors='coerce')
                cutoff_dt = pd.Timestamp.utcnow() - delta
                mask = dt >= cutoff_dt
                return df.loc[mask]
            except Exception:
                pass

    # 3) Fallback: row-count based on filename interval
    sec = infer_candle_seconds_from_filename(file_path) or 3600  # assume 1h if unknown
    need = int(max(warmup + 1, delta.total_seconds() // sec))
    if need <= 0 or need >= len(df):
        return df
    return df.iloc[-need:]

# ===================== Main =====================
def run_simulation(*, timeframe: str = "1m", brain_name: str = "exhaustion", viz: bool = True) -> None:
    # Load hourly candles
    file_path = "data/sim/SOLUSD_1h.csv"
    df = pd.read_csv(file_path)

    brain = load_brain(brain_name)

    # Robust timeframe handling (works even without a timestamp column)
    delta = parse_timeframe(timeframe)
    if delta is not None:
        df = apply_time_filter(df, delta, file_path, brain.warmup())

    df = df.reset_index(drop=True)
    df["candle_index"] = range(len(df))

    brain.prepare(df)

    if viz:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(df["candle_index"], df["close"], lw=1, label="Close Price", color="blue")

    for t in range(brain.warmup(), len(df), WINDOW_STEP):
        brain.step(df, t)

    pts = brain.overlays()

    if not viz:
        return

    # ===================== Plot & Toggles (lazy create) =====================
    ax1.set_title("Price with Exhaustion + Predictors (Keys 1–2,3,4–8; Letters W/E/R/T)")
    ax1.set_xlabel("Candles (Index)")
    ax1.set_ylabel("Price")
    ax1.grid(True)

    artists = {
        "exhaustion": None,
        "reversals":  None,
        "bottom4":    None,
        "top5":       None,
        "top6":       None,
        "top7":       None,
        "top8":       None,
        "valley_w":   None,
        "valley_e":   None,
        "valley_r":   None,
        "valley_t":   None,
    }

    state = {k: False for k in artists.keys()}

    def ensure_artist(name: str):
        if artists[name] is not None:
            return
        if name == "exhaustion":
            # combine red/green
            xr, yr, sr = pts["exhaustion_red"]["x"], pts["exhaustion_red"]["y"], pts["exhaustion_red"]["s"]
            xg, yg, sg = pts["exhaustion_green"]["x"], pts["exhaustion_green"]["y"], pts["exhaustion_green"]["s"]
            h1 = ax1.scatter(xr, yr, s=sr, c="red", zorder=6, visible=False)
            h2 = ax1.scatter(xg, yg, s=sg, c="green", zorder=6, visible=False)
            artists[name] = (h1, h2)
        elif name == "reversals":
            artists[name] = ax1.scatter(pts["reversal"]["x"], pts["reversal"]["y"],
                                        c="yellow", s=120, edgecolor="black", zorder=7, visible=False)
        elif name in ("bottom4","top5","top6","top7","top8","valley_w","valley_e","valley_r","valley_t"):
            style = {
                "bottom4": dict(c="cyan", marker="v", s=100, zorder=6),
                "top5":    dict(c="orange", marker="s", s=110, zorder=6),
                "top6":    dict(c="red", marker="*", s=140, zorder=6),
                "top7":    dict(c="purple", marker="^", s=110, zorder=6),
                "top8":    dict(c="magenta", marker="P", s=160, zorder=7),
                "valley_w":dict(c="teal", marker="h", s=120, zorder=7),
                "valley_e":dict(c="deepskyblue", marker="D", s=110, zorder=7),
                "valley_r":dict(c="darkcyan", marker="s", s=100, zorder=7),
                "valley_t":dict(c="turquoise", marker="P", s=150, zorder=8),
            }[name]
            artists[name] = ax1.scatter(pts[name]["x"], pts[name]["y"], visible=False, **style)

    def set_visible(name: str, on: bool):
        h = artists[name]
        if h is None:
            return
        if isinstance(h, tuple):
            for hh in h:
                hh.set_visible(on)
        else:
            h.set_visible(on)

    def toggle(name: str):
        ensure_artist(name)
        state[name] = not state[name]
        set_visible(name, state[name])
        print(f"[TOGGLE] {name} {'ON' if state[name] else 'OFF'}")
        plt.draw()

    def on_key(event):
        k = (event.key or "").lower()
        if k == "1":
            toggle("exhaustion")
        elif k == "2":
            toggle("reversals")
        elif k == "3" or k == "r":
            toggle("valley_r")
        elif k == "4":
            toggle("bottom4")
        elif k == "5":
            toggle("top5")
        elif k == "6":
            toggle("top6")
        elif k == "7":
            toggle("top7")
        elif k == "8":
            toggle("top8")
        elif k == "w":
            toggle("valley_w")
        elif k == "e":
            toggle("valley_e")
        elif k == "t":
            toggle("valley_t")
        # ignore q to avoid closing figure

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--time",
        type=str,
        default="1m",
        help="Time window (e.g. '3m' for three months)",
    )
    p.add_argument("--brain", type=str, default="exhaustion")
    p.add_argument("--viz", action="store_true")
    args = p.parse_args()
    run_simulation(timeframe=args.time, brain_name=args.brain, viz=args.viz)


if __name__ == "__main__":
    main()
