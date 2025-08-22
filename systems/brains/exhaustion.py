from __future__ import annotations

from collections import deque, defaultdict

import numpy as np
import pandas as pd

from .base import Brain

# ===================== Parameters =====================
WINDOW_SIZE = 24        # candles per box (for exhaustion voting)
WINDOW_STEP = 2         # step between boxes
CLUSTER_WINDOW = 10     # lookback to count exhaustion density
BASE_SIZE = 10          # base dot size
SCALE_POWER = 2         # exhaustion growth scale


# ===================== Helpers =====================
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


class ExhaustionBrain(Brain):
    name = "exhaustion"

    def __init__(self):
        self._pts = {
            "exhaustion_red":  {"x": [], "y": [], "s": [], "c": "red"},
            "exhaustion_green":{"x": [], "y": [], "s": [], "c": "green"},
            "reversal":        {"x": [], "y": []},
            "bottom4":         {"x": [], "y": []},
            "top5":            {"x": [], "y": []},
            "top6":            {"x": [], "y": []},
            "top7":            {"x": [], "y": []},
            "top8":            {"x": [], "y": []},
            "valley_w":        {"x": [], "y": []},
            "valley_e":        {"x": [], "y": []},
            "valley_r":        {"x": [], "y": []},
            "valley_t":        {"x": [], "y": []},
        }
        self._last_exh: int | None = None
        self._recent_buys = deque(maxlen=CLUSTER_WINDOW)
        self._recent_sells = deque(maxlen=CLUSTER_WINDOW)
        self._last_idx = defaultdict(lambda: -10)
        self._have_hl = False
        self._meta_done = False
        self._cluster_events: list[tuple[int, int, str]] = []  # (t, strength, color)

    def warmup(self) -> int:
        return WINDOW_SIZE - 1

    def prepare(self, df):
        self._have_hl = all(col in df.columns for col in ["high", "low", "close"])
        if self._have_hl:
            prev_close = df["close"].shift(1)
            tr = pd.concat([
                (df["high"] - df["low"]).abs(),
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ], axis=1).max(axis=1)
            df["atr14"] = tr.rolling(14, min_periods=1).mean()
        else:
            df["atr14"] = df["close"].rolling(14, min_periods=1).std()
        roll_mean = df["close"].rolling(50, min_periods=1).mean()
        roll_std = df["close"].rolling(50, min_periods=1).std().replace(0, 1)
        df["z50"] = (df["close"] - roll_mean) / roll_std

    def step(self, df, t: int) -> None:
        candle = df.iloc[t]
        x = int(candle["candle_index"])
        y = float(candle["close"])

        decision, confidence, score = multi_window_vote(df, t, window_sizes=[8, 12, 24, 48])

        # Key 1: Exhaustion (red/green clusters)
        if decision == 1:  # SELL exhaustion
            self._recent_buys.append(t)
            cluster_strength = sum(1 for idx in self._recent_buys if t - idx <= CLUSTER_WINDOW)
            size = BASE_SIZE * (cluster_strength ** SCALE_POWER)
            self._pts["exhaustion_red"]["x"].append(x)
            self._pts["exhaustion_red"]["y"].append(y)
            self._pts["exhaustion_red"]["s"].append(size)
            self._cluster_events.append((t, cluster_strength, "red"))
        elif decision == -1:  # BUY exhaustion
            self._recent_sells.append(t)
            cluster_strength = sum(1 for idx in self._recent_sells if t - idx <= CLUSTER_WINDOW)
            size = BASE_SIZE * (cluster_strength ** SCALE_POWER)
            self._pts["exhaustion_green"]["x"].append(x)
            self._pts["exhaustion_green"]["y"].append(y)
            self._pts["exhaustion_green"]["s"].append(size)
            self._cluster_events.append((t, cluster_strength, "green"))

        # Key 2: Reversals (yellow) on color flip
        if self._last_exh is not None and decision != 0 and decision != self._last_exh:
            self._pts["reversal"]["x"].append(x)
            self._pts["reversal"]["y"].append(y)
        if decision != 0:
            self._last_exh = decision

        # Slopes
        slope_now = 0.0
        slope_prev = 0.0
        if t >= 48:
            sub_now = df["close"].iloc[t - 24:t]
            sub_prev = df["close"].iloc[t - 48:t - 24]
            slope_now = float(np.polyfit(np.arange(len(sub_now)), sub_now, 1)[0]) if len(sub_now) > 1 else 0.0
            slope_prev = float(np.polyfit(np.arange(len(sub_prev)), sub_prev, 1)[0]) if len(sub_prev) > 1 else 0.0

        # Key 4: Bottom catcher (local min + improving slope)
        if t >= 36:
            lookback = 12
            window = df["close"].iloc[t - lookback:t + 1]
            if y == float(window.min()) and slope_now > slope_prev:
                self._pts["bottom4"]["x"].append(x)
                self._pts["bottom4"]["y"].append(y)

        # Key 5: Divergence (bearish)
        if t >= 48:
            price_now = df["close"].iloc[t - 1]
            price_prev = df["close"].iloc[t - 25]
            if price_now > price_prev and slope_now < slope_prev:
                self._pts["top5"]["x"].append(x)
                self._pts["top5"]["y"].append(y)

        # Key 6: Rolling Peak Detection (swing-high)
        if t >= 12:
            lookback = 12
            win = df["close"].iloc[t - lookback:t + 1]
            if y == float(win.max()):
                self._pts["top6"]["x"].append(x)
                self._pts["top6"]["y"].append(y)

        # Key 7: Compression -> Expansion at highs (predictive)
        if t >= 18:
            rng_window = 12
            sub_rng = df["close"].iloc[t - rng_window:t]
            rng = float(sub_rng.max() - sub_rng.min())
            meanv = float(sub_rng.mean()) if float(sub_rng.mean()) != 0 else 1.0
            if rng < 0.02 * meanv and decision == 1:
                self._pts["top7"]["x"].append(x)
                self._pts["top7"]["y"].append(y)

        # ----- Valley Detectors -----
        # W: Valley A - Capitulation Wick + Snap (tuned; fires a bit more)
        if self._have_hl and t >= 2:
            o = df["open"].iloc[t] if "open" in df.columns else df["close"].iloc[t - 1]
            h_ = df["high"].iloc[t] if "high" in df.columns else max(y, o)
            l_ = df["low"].iloc[t] if "low" in df.columns else min(y, o)
            rng = max(h_ - l_, 1e-9)
            lower_wick = (y - l_) / rng if y >= l_ else 0.0
            body = abs(y - o) / rng
            atr_now = float(df["atr14"].iloc[t])
            atr_hist = df["atr14"].iloc[max(0, t - 200):t + 1]
            z = float(df["z50"].iloc[t])
            wick_ok = lower_wick >= 0.50
            body_ok = body <= 0.55
            atr_ok = atr_now >= np.nanpercentile(atr_hist, 75)
            z_ok = z <= -0.3
            if wick_ok and body_ok and atr_ok and z_ok:
                if t + 1 < len(df):
                    next_close = float(df["close"].iloc[t + 1])
                    mid_body = (o + y) / 2.0
                    if (next_close > mid_body) or (next_close > y):
                        self._pts["valley_w"]["x"].append(x)
                        self._pts["valley_w"]["y"].append(y)
                        self._last_idx["valley_w"] = x

        # E: Valley B - Exhaustion Super-Cluster + Positive Divergence (tuned livelier)
        if t >= 36:
            N = 10
            x_cut = x - N
            recent_exh = 0
            for arr in ("exhaustion_red", "exhaustion_green"):
                xs = self._pts[arr]["x"]
                if xs:
                    i = len(xs) - 1
                    while i >= 0 and xs[i] >= x_cut:
                        recent_exh += 1
                        i -= 1
            z = float(df["z50"].iloc[t])
            div_ok = slope_now >= slope_prev
            guard = z <= -0.5
            extra = False
            if t >= 8:
                win9 = df["close"].iloc[t - 8:t + 1]
                extra = y == float(win9.min())
            if (t + 1 < len(df)) and not extra:
                extra = float(df["close"].iloc[t + 1]) > y
            if recent_exh >= 2 and div_ok and guard and extra:
                self._pts["valley_e"]["x"].append(x)
                self._pts["valley_e"]["y"].append(y)
                self._last_idx["valley_e"] = x

        # R: Valley D - Drawdown Z-score + Reversion (relaxed) -- also mapped to numeric key '3'
        if t >= 50:
            z = float(df["z50"].iloc[t])
            if z <= -1.7:
                sub3 = df["close"].iloc[t - 3:t]
                slope_ok = False
                if len(sub3) > 1:
                    slope3 = float(np.polyfit(np.arange(len(sub3)), sub3, 1)[0])
                    slope_ok = slope3 > 0
                next_ok = (t + 1 < len(df)) and (float(df["close"].iloc[t + 1]) > y)
                if slope_ok or next_ok:
                    self._pts["valley_r"]["x"].append(x)
                    self._pts["valley_r"]["y"].append(y)
                    self._last_idx["valley_r"] = x

        # T: Confluence - >=2 valley signals within 2 bars AND local 7-bar min
        if t >= 6:
            win7 = df["close"].iloc[t - 6:t + 1]
            is_local_min = y == float(win7.min())
        else:
            is_local_min = False
        near_count = sum(1 for k in ("valley_w", "valley_e", "valley_r") if abs(x - self._last_idx[k]) <= 2)
        if is_local_min and near_count >= 2:
            self._pts["valley_t"]["x"].append(x)
            self._pts["valley_t"]["y"].append(y)

    def overlays(self):
        if not self._meta_done and self._pts["reversal"]["x"] and self._pts["top5"]["x"]:
            i, j = 0, 0
            Rx, Ry = self._pts["reversal"]["x"], self._pts["reversal"]["y"]
            Dx, Dy = self._pts["top5"]["x"], self._pts["top5"]["y"]
            while i < len(Rx) and j < len(Dx):
                if abs(Rx[i] - Dx[j]) <= 2:
                    self._pts["top8"]["x"].append(Rx[i])
                    self._pts["top8"]["y"].append(Ry[i])
                    i += 1
                    j += 1
                elif Rx[i] < Dx[j]:
                    i += 1
                else:
                    j += 1
            self._meta_done = True
        return self._pts

    # ===================== Stats =====================
    def compute_stats(
        self, df: pd.DataFrame, trend_state: list[int], slopes: list[float]
    ) -> dict[str, float]:
        TREND_MIN_LEN = 50
        POST_FLIP_WINDOW = 92

        n = len(trend_state)
        segments: list[tuple[int, int, int]] = []  # (dir, start, end)
        if n:
            start = 0
            cur = trend_state[0]
            for i in range(1, n):
                if trend_state[i] != cur:
                    segments.append((cur, start, i - 1))
                    start = i
                    cur = trend_state[i]
            segments.append((cur, start, n - 1))

        up_durations, down_durations = [], []
        up_bubbles, down_bubbles = [], []

        for dir_, s, e in segments:
            length = e - s + 1
            if dir_ == 1:
                if length >= TREND_MIN_LEN:
                    up_durations.append(length - TREND_MIN_LEN)
                max_strength = 0
                for t, strength, color in self._cluster_events:
                    if s <= t <= e and color == "red" and strength > max_strength:
                        max_strength = strength
                up_bubbles.append(max_strength)
            elif dir_ == -1:
                if length >= TREND_MIN_LEN:
                    down_durations.append(length - TREND_MIN_LEN)
                max_strength = 0
                for t, strength, color in self._cluster_events:
                    if s <= t <= e and color == "green" and strength > max_strength:
                        max_strength = strength
                down_bubbles.append(max_strength)

        def _avg(vals):
            return float(np.mean(vals)) if vals else 0.0

        angles = slopes
        down_deltas, up_deltas = [], []
        for i in range(1, n):
            if trend_state[i - 1] == -1 and trend_state[i] != -1:
                angle_flip = angles[i]
                window = angles[i + 1 : i + 1 + POST_FLIP_WINDOW]
                if len(window) > 0:
                    down_deltas.append(float(np.mean([a - angle_flip for a in window])))
            if trend_state[i - 1] == 1 and trend_state[i] != 1:
                angle_flip = angles[i]
                window = angles[i + 1 : i + 1 + POST_FLIP_WINDOW]
                if len(window) > 0:
                    up_deltas.append(float(np.mean([a - angle_flip for a in window])))

        stats = {
            "avg_uptrend_duration_past50": _avg(up_durations),
            "avg_downtrend_duration_past50": _avg(down_durations),
            "avg_down_slope_angle_delta_post_flip_92": _avg(down_deltas),
            "avg_up_slope_angle_delta_post_flip_92": _avg(up_deltas),
            "avg_max_pressure_up_bubble": _avg(up_bubbles),
            "avg_max_pressure_down_bubble": _avg(down_bubbles),
        }
        return stats
