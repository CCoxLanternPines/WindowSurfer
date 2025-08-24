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
    decisions_buy, decisions_sell, decisions_hold = [], [], []

    pts = {
        "a_up": {"x": [], "y": [], "s": []},
        "a_down": {"x": [], "y": [], "s": []},
    }

    current_trend = "neutral"
    pressure_up = 0
    pressure_down = 0

    up_keys = {"1", "5", "6", "7", "8"}
    down_keys = {"w", "e", "r", "t", "y", "3", "4"}

    for t in range(50, len(df)):
        features = extract_features_at_t(brain_cache, t)
        decision, reasons, score, feat_snapshot = run_arbiter(
            features, state, debug=True, return_score=True
        )

        x = int(df["candle_index"].iloc[t])
        y = float(df["close"].iloc[t])
        payload = (decision, x, y, reasons, score, feat_snapshot)

        if decision == "BUY":
            decisions_buy.append(payload)
            buy_signals.append((x, y))
            if current_trend == "down":
                size = 200 + pressure_down * 50
                pts["a_up"]["x"].append(x)
                pts["a_up"]["y"].append(y)
                pts["a_up"]["s"].append(size)
                pressure_down = 0
                pressure_up = 1
            else:
                pressure_up += 1
            current_trend = "up"
            state = "long"
        elif decision == "SELL":
            decisions_sell.append(payload)
            sell_signals.append((x, y))
            if current_trend == "up":
                size = 200 + pressure_up * 50
                pts["a_down"]["x"].append(x)
                pts["a_down"]["y"].append(y)
                pts["a_down"]["s"].append(size)
                pressure_up = 0
                pressure_down = 1
            else:
                pressure_down += 1
            current_trend = "down"
            state = "flat"
        else:  # HOLD
            if t % 10 == 0:
                decisions_hold.append(payload)

    if viz:
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots(figsize=(12, 6))
        fig.tight_layout()
        fig.subplots_adjust(top=0.96, bottom=0.065, left=0.06, right=0.775)
        ax1.plot(df["candle_index"], df["close"], lw=1, color="blue")

        scatter_buy = ax1.scatter(
            [d[1] for d in decisions_buy],
            [d[2] for d in decisions_buy],
            color="green",
            marker="^",
            s=120,
            label="BUY",
            picker=True,
        )
        scatter_sell = ax1.scatter(
            [d[1] for d in decisions_sell],
            [d[2] for d in decisions_sell],
            color="red",
            marker="v",
            s=120,
            label="SELL",
            picker=True,
        )
        scatter_hold = ax1.scatter(
            [d[1] for d in decisions_hold],
            [d[2] for d in decisions_hold],
            color="gray",
            marker="o",
            s=40,
            alpha=0.3,
            label="HOLD",
            picker=True,
        )
        ax1.legend()

        info_box = ax1.text(
            1.02,
            0.95,
            "Click a marker",
            transform=ax1.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.7),
        )

        artists: dict[str, object] = {}

        def ensure_artist(name: str):
            if name not in artists:
                if name == "a_up":
                    artists[name] = ax1.scatter(
                        pts["a_up"]["x"],
                        pts["a_up"]["y"],
                        marker="^",
                        c="green",
                        s=pts["a_up"]["s"],
                        zorder=9,
                        visible=False,
                    )
                elif name == "a_down":
                    artists[name] = ax1.scatter(
                        pts["a_down"]["x"],
                        pts["a_down"]["y"],
                        marker="v",
                        c="red",
                        s=pts["a_down"]["s"],
                        zorder=9,
                        visible=False,
                    )
            return artists.get(name)

        def toggle(name: str):
            art = ensure_artist(name)
            if art:
                art.set_visible(not art.get_visible())
                fig.canvas.draw_idle()

        def on_pick(event):
            ind = event.ind[0]
            artist = event.artist

            if artist == scatter_buy:
                d = decisions_buy[ind]
            elif artist == scatter_sell:
                d = decisions_sell[ind]
            else:
                d = decisions_hold[ind]

            decision, x, y, reasons, score, feats = d
            feat_lines = [f"{k}={v}" for k, v in feats.items()]
            if len(feat_lines) > 20:
                remaining = len(feat_lines) - 20
                feat_lines = feat_lines[:20] + [f"... (+{remaining} more)"]

            info_box.set_text(
                f"{decision} @ idx={x} price={y:.2f}\n"
                f"Score={score:+.3f}\n\n"
                "[ARB CHECKS]\n" + "\n".join(reasons) + "\n\n"
                "[FEATURES]\n" + "\n".join(feat_lines)
            )
            fig.canvas.draw_idle()

        def on_key(event):
            k = event.key
            if k == "a":
                toggle("a_up")
                toggle("a_down")

        fig.canvas.mpl_connect("pick_event", on_pick)
        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.show()

    print(f"[SIM][{timeframe}] buys={len(buy_signals)} sells={len(sell_signals)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--time", default="1m")
    parser.add_argument("--viz", action="store_true")
    args = parser.parse_args()
    run_simulation(timeframe=args.time, viz=args.viz)
