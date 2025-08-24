
from __future__ import annotations

import argparse
import re
from datetime import timedelta, datetime, timezone
from collections import deque, defaultdict
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===================== Parameters =====================
WINDOW_SIZE   = 24        # candles per box (for exhaustion voting)
WINDOW_STEP   = 2         # step between boxes
CLUSTER_WINDOW= 10        # lookback to count exhaustion density
BASE_SIZE     = 10        # base dot size
SCALE_POWER   = 2         # exhaustion growth scale

# ===================== Helpers =====================
_INTERVAL_RE = re.compile(r'[_\-]((\d+)([smhdw]))(?=\.|_|$)', re.I)

# Seconds mapping for user-facing timeframes where ``m`` means months.
TIMEFRAME_SECONDS = {
    's': 1,
    'm': 30 * 24 * 3600,  # month (≈30 days)
    'h': 3600,
    'd': 86400,
    'w': 604800,
}

# Separate mapping for candle intervals in filenames where ``m`` means minutes.
INTERVAL_SECONDS = {
    's': 1,
    'm': 60,
    'h': 3600,
    'd': 86400,
    'w': 604800,
}

def parse_timeframe(tf: str) -> timedelta | None:
    """Parse strings like '12h', '3d', '1m', '6w' into ``timedelta``.

    "m" is interpreted as months (approximately 30 days).
    """
    if not tf:
        return None
    m = re.match(r'(?i)^\s*(\d+)\s*([smhdw])\s*$', tf)
    if not m:
        return None
    n, u = int(m.group(1)), m.group(2).lower()
    return timedelta(seconds=n * TIMEFRAME_SECONDS[u])

def infer_candle_seconds_from_filename(path: str) -> int | None:
    """Try to infer candle interval from filename like *_1h.csv, *_15m.csv, *_1d.csv."""
    m = _INTERVAL_RE.search(os.path.basename(path))
    if not m:
        return None
    n, u = int(m.group(2)), m.group(3).lower()
    return n * INTERVAL_SECONDS[u]

def apply_time_filter(df: pd.DataFrame, delta: timedelta, file_path: str) -> pd.DataFrame:
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
    need = int(max(WINDOW_SIZE, delta.total_seconds() // sec))
    if need <= 0 or need >= len(df):
        return df
    return df.iloc[-need:]

def multi_window_vote(df, t, window_sizes, slope_thresh=0.001, range_thresh=0.05):
    """Return (-1,0,1) decision with confidence using multi-window slope direction."""
    votes, strengths = [], []
    for W in window_sizes:
        if t - W < 0:
            continue
        sub = df.iloc[t-W:t]
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

# ===================== Main =====================
def run_simulation(*, timeframe: str = "1m", viz: bool = True) -> None:
    # Load hourly candles
    file_path = "data/sim/SOLUSD_1h.csv"
    df = pd.read_csv(file_path)

    # Robust timeframe handling (works even without a timestamp column)
    delta = parse_timeframe(timeframe)
    if delta is not None:
        df = apply_time_filter(df, delta, file_path)

    df = df.reset_index(drop=True)
    df["candle_index"] = range(len(df))

    # ----- Precompute ATR(14) and Z-score(50) -----
    have_hl = all(col in df.columns for col in ["high", "low", "close"])
    if have_hl:
        prev_close = df["close"].shift(1)
        tr = pd.concat([
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs()
        ], axis=1).max(axis=1)
        df["atr14"] = tr.rolling(14, min_periods=1).mean()
    else:
        df["atr14"] = df["close"].rolling(14, min_periods=1).std()

    roll_mean = df["close"].rolling(50, min_periods=1).mean()
    roll_std  = df["close"].rolling(50, min_periods=1).std().replace(0, 1)
    df["z50"] = (df["close"] - roll_mean) / roll_std

    if viz:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(df["candle_index"], df["close"], lw=1, label="Close Price", color="blue")

    # ----- Lazy-plot storage -----
    pts = {
        "exhaustion_red":  {"x": [], "y": [], "s": [], "c": "red"},
        "exhaustion_green":{"x": [], "y": [], "s": [], "c": "green"},
        "reversal":        {"x": [], "y": []},

        "bottom4":         {"x": [], "y": []},
        "top5":            {"x": [], "y": []},
        "top6":            {"x": [], "y": []},
        "top7":            {"x": [], "y": []},
        "top8":            {"x": [], "y": []},

        # Valleys
        "valley_w":        {"x": [], "y": []},   # Wick+Snap (W)
        "valley_e":        {"x": [], "y": []},   # Exhaustion+Div (E)
        "valley_r":        {"x": [], "y": []},   # Drawdown Z + Reversion (R / 3)
        "valley_t":        {"x": [], "y": []},   # Confluence (T)


        "pressure_a_top":    {"x": [], "y": [], "s": []},
        "pressure_a_bottom": {"x": [], "y": [], "s": []},

    }

    last_exhaustion_decision: int | None = None
    recent_buys  = deque(maxlen=CLUSTER_WINDOW)
    recent_sells = deque(maxlen=CLUSTER_WINDOW)

    # For confluence tracking, keep last indices
    last_idx = defaultdict(lambda: -10)

    trend_data = []

    # ----- Iterate bars (lightweight, store only coords) -----
    for t in range(WINDOW_SIZE - 1, len(df), WINDOW_STEP):
        candle = df.iloc[t]
        x = int(candle["candle_index"])
        y = float(candle["close"])

        decision, confidence, score = multi_window_vote(df, t, window_sizes=[8, 12, 24, 48])
        trend_data.append((x, y, decision))

        # Key 1: Exhaustion (red/green clusters)
        if decision == 1:  # SELL exhaustion
            recent_buys.append(t)
            cluster_strength = sum(1 for idx in recent_buys if t - idx <= CLUSTER_WINDOW)
            size = BASE_SIZE * (cluster_strength ** SCALE_POWER)
            pts["exhaustion_red"]["x"].append(x)
            pts["exhaustion_red"]["y"].append(y)
            pts["exhaustion_red"]["s"].append(size)
        elif decision == -1:  # BUY exhaustion
            recent_sells.append(t)
            cluster_strength = sum(1 for idx in recent_sells if t - idx <= CLUSTER_WINDOW)
            size = BASE_SIZE * (cluster_strength ** SCALE_POWER)
            pts["exhaustion_green"]["x"].append(x)
            pts["exhaustion_green"]["y"].append(y)
            pts["exhaustion_green"]["s"].append(size)

        # Key 2: Reversals (yellow) on color flip
        if last_exhaustion_decision is not None and decision != 0 and decision != last_exhaustion_decision:
            pts["reversal"]["x"].append(x)
            pts["reversal"]["y"].append(y)
        if decision != 0:
            last_exhaustion_decision = decision

        # Slopes
        slope_now = 0.0
        slope_prev = 0.0
        if t >= 48:
            sub_now  = df["close"].iloc[t-24:t]
            sub_prev = df["close"].iloc[t-48:t-24]
            slope_now  = float(np.polyfit(np.arange(len(sub_now)),  sub_now,  1)[0]) if len(sub_now)  > 1 else 0.0
            slope_prev = float(np.polyfit(np.arange(len(sub_prev)), sub_prev, 1)[0]) if len(sub_prev) > 1 else 0.0

        # Key 4: Bottom catcher (local min + improving slope)
        if t >= 36:
            lookback = 12
            window = df["close"].iloc[t-lookback:t+1]
            if y == float(window.min()) and slope_now > slope_prev:
                pts["bottom4"]["x"].append(x)
                pts["bottom4"]["y"].append(y)

        # Key 5: Divergence (bearish)
        if t >= 48:
            price_now  = df["close"].iloc[t-1]
            price_prev = df["close"].iloc[t-25]
            if price_now > price_prev and slope_now < slope_prev:
                pts["top5"]["x"].append(x)
                pts["top5"]["y"].append(y)

        # Key 6: Rolling Peak Detection (swing-high)
        if t >= 12:
            lookback = 12
            win = df["close"].iloc[t-lookback:t+1]
            if y == float(win.max()):
                pts["top6"]["x"].append(x)
                pts["top6"]["y"].append(y)

        # Key 7: Compression -> Expansion at highs (predictive)
        if t >= 18:
            rng_window = 12
            sub_rng = df["close"].iloc[t-rng_window:t]
            rng = float(sub_rng.max() - sub_rng.min())
            meanv = float(sub_rng.mean()) if float(sub_rng.mean()) != 0 else 1.0
            if rng < 0.02 * meanv and decision == 1:
                pts["top7"]["x"].append(x)
                pts["top7"]["y"].append(y)

        # ----- Valley Detectors -----
        # W: Valley A - Capitulation Wick + Snap (tuned; fires a bit more)
        if have_hl and t >= 2:
            # If 'open' not present, approximate with prior close
            o = df["open"].iloc[t] if "open" in df.columns else df["close"].iloc[t-1]
            h_ = df["high"].iloc[t] if "high" in df.columns else max(y, o)
            l_ = df["low"].iloc[t]  if "low"  in df.columns else min(y, o)
            rng = max(h_ - l_, 1e-9)
            lower_wick = (y - l_) / rng if y >= l_ else 0.0
            body = abs(y - o) / rng
            atr_now = float(df["atr14"].iloc[t])
            atr_hist = df["atr14"].iloc[max(0, t-200):t+1]
            z = float(df["z50"].iloc[t])
            wick_ok = lower_wick >= 0.50
            body_ok = body <= 0.55
            atr_ok  = atr_now >= np.nanpercentile(atr_hist, 75)
            z_ok    = z <= -0.3
            if wick_ok and body_ok and atr_ok and z_ok:
                if t+1 < len(df):
                    next_close = float(df["close"].iloc[t+1])
                    mid_body = (o + y)/2.0
                    if (next_close > mid_body) or (next_close > y):
                        pts["valley_w"]["x"].append(x); pts["valley_w"]["y"].append(y); last_idx["valley_w"] = x

        # E: Valley B - Exhaustion Super-Cluster + Positive Divergence (tuned livelier)
        if t >= 36:
            N = 10
            x_cut = x - N
            # count exhaustion points in last N bars (use red/green lists)
            recent_exh = 0
            for arr in ("exhaustion_red","exhaustion_green"):
                xs = pts[arr]["x"]
                if xs:
                    i = len(xs) - 1
                    while i >= 0 and xs[i] >= x_cut:
                        recent_exh += 1
                        i -= 1
            z = float(df["z50"].iloc[t])
            div_ok = (slope_now >= slope_prev)  # not deteriorating
            guard  = (z <= -0.5)
            extra = False
            if t >= 8:
                win9 = df["close"].iloc[t-8:t+1]
                extra = (y == float(win9.min()))
            if (t+1 < len(df)) and not extra:
                extra = float(df["close"].iloc[t+1]) > y

            if recent_exh >= 2 and div_ok and guard and extra:
                pts["valley_e"]["x"].append(x); pts["valley_e"]["y"].append(y); last_idx["valley_e"] = x

        # R: Valley D - Drawdown Z-score + Reversion (relaxed)  -- also mapped to numeric key '3'
        if t >= 50:
            z = float(df["z50"].iloc[t])
            if z <= -1.7:
                sub3 = df["close"].iloc[t-3:t]
                slope_ok = False
                if len(sub3) > 1:
                    slope3 = float(np.polyfit(np.arange(len(sub3)), sub3, 1)[0])
                    slope_ok = slope3 > 0
                next_ok = (t+1 < len(df)) and (float(df["close"].iloc[t+1]) > y)
                if slope_ok or next_ok:
                    pts["valley_r"]["x"].append(x); pts["valley_r"]["y"].append(y); last_idx["valley_r"] = x

        # T: Confluence - >=2 valley signals within 2 bars AND local 7-bar min
        if t >= 6:
            win7 = df["close"].iloc[t-6:t+1]
            is_local_min = y == float(win7.min())
        else:
            is_local_min = False

        # count how many of W/E/R last indices are "near" x (within 2)
        near_count = sum(1 for k in ("valley_w","valley_e","valley_r") if abs(x - last_idx[k]) <= 2)
        if is_local_min and near_count >= 2:
            pts["valley_t"]["x"].append(x); pts["valley_t"]["y"].append(y)

    current_trend = None
    pressure_counter = 0
    for x, y, decision in trend_data:

        trend = "up" if decision == 1 else "down" if decision == -1 else None
        if trend is None:

            continue
        if current_trend is None:
            current_trend = trend
            pressure_counter = 1
        elif trend == current_trend:
            pressure_counter += 1
        else:

            size = BASE_SIZE * (pressure_counter ** 2) * 0.5
            if current_trend == "up" and trend == "down":
                pts["pressure_a_top"]["x"].append(x)
                pts["pressure_a_top"]["y"].append(y)
                pts["pressure_a_top"]["s"].append(size)
            elif current_trend == "down" and trend == "up":
                pts["pressure_a_bottom"]["x"].append(x)
                pts["pressure_a_bottom"]["y"].append(y)
                pts["pressure_a_bottom"]["s"].append(size)

            current_trend = trend
            pressure_counter = 1

    # Key 8: Meta-Filter (rev + div overlap) computed post-loop by proximity
    if pts["reversal"]["x"] and pts["top5"]["x"]:
        i, j = 0, 0
        Rx, Ry = pts["reversal"]["x"], pts["reversal"]["y"]
        Dx, Dy = pts["top5"]["x"],    pts["top5"]["y"]
        while i < len(Rx) and j < len(Dx):
            if abs(Rx[i] - Dx[j]) <= 2:
                pts["top8"]["x"].append(Rx[i]); pts["top8"]["y"].append(Ry[i])
                i += 1; j += 1
            elif Rx[i] < Dx[j]:
                i += 1
            else:
                j += 1

    if not viz:
        return

    # ===================== Plot & Toggles (lazy create) =====================
    ax1.set_title("Price with Exhaustion + Predictors (Keys 1/u,2,3,4–8; Letters W/E/R/T)")
    ax1.set_xlabel("Candles (Index)")
    ax1.set_ylabel("Price")
    ax1.grid(True)

    artists = {
        "exhaustion_up":   None,  # red SELL exhaustion
        "exhaustion_down": None,  # green BUY exhaustion
        "reversals":       None,
        "bottom4":         None,
        "top5":            None,
        "top6":            None,
        "top7":            None,
        "top8":            None,
        "valley_w":        None,
        "valley_e":        None,
        "valley_r":        None,
        "valley_t":        None,
        "pressure_a":      None,
    }

    state = {k: False for k in artists.keys()}

    def ensure_artist(name: str):
        if artists[name] is not None:
            return
        if name == "exhaustion_up":
            xr, yr, sr = pts["exhaustion_red"]["x"], pts["exhaustion_red"]["y"], pts["exhaustion_red"]["s"]
            artists[name] = ax1.scatter(xr, yr, s=sr, c="red", zorder=6, visible=False)
        elif name == "exhaustion_down":
            xg, yg, sg = pts["exhaustion_green"]["x"], pts["exhaustion_green"]["y"], pts["exhaustion_green"]["s"]
            artists[name] = ax1.scatter(xg, yg, s=sg, c="green", zorder=6, visible=False)
        elif name == "reversals":
            artists[name] = ax1.scatter(pts["reversal"]["x"], pts["reversal"]["y"],
                                        c="yellow", s=120, edgecolor="black", zorder=7, visible=False)
        elif name == "pressure_a":

            scat_top = ax1.scatter(
                pts["pressure_a_top"]["x"],
                pts["pressure_a_top"]["y"],
                s=pts["pressure_a_top"]["s"],
                c="gray",
                alpha=1,
                marker="o",
                zorder=6,
                visible=False,
            )
            scat_bottom = ax1.scatter(
                pts["pressure_a_bottom"]["x"],
                pts["pressure_a_bottom"]["y"],
                s=pts["pressure_a_bottom"]["s"],
                c="black",
                alpha=1,
                marker="s",
                zorder=6,
                visible=False,
            )
            artists[name] = (scat_top, scat_bottom)

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
            toggle("exhaustion_up")
        elif k == "u":
            toggle("exhaustion_down")
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
        elif k == "a":
            toggle("pressure_a")
        # ignore q to avoid closing figure

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    plt.show()

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--time",
        type=str,
        default="1m",
        help="Simulation window (e.g., 1m for one month)",
    )
    p.add_argument("--viz", action="store_true")
    args = p.parse_args()
    run_simulation(timeframe=args.time, viz=args.viz)

if __name__ == "__main__":
    main()
