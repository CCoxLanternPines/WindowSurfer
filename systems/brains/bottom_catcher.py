from __future__ import annotations

"""Brain 4: Bottom catcher with valley-quality statistics."""

from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..sim_engine import WINDOW_SIZE, WINDOW_STEP


ALIGN_WINDOW = 5  # ±5 candles for extrema alignment
SLOPE_DELTA_THRESH = 0.002
ATR_MED_WINDOW = 100


def run(df: pd.DataFrame, viz: bool):
    """Detect local bottoms with improving slope."""
    signals: List[Dict[str, float]] = []
    xs, ys = [], []

    slope_now = 0.0
    slope_prev = 0.0

    for t in range(WINDOW_SIZE - 1, len(df), WINDOW_STEP):
        if t >= 48:
            sub_now = df["close"].iloc[t-24:t]
            sub_prev = df["close"].iloc[t-48:t-24]
            slope_now = float(np.polyfit(np.arange(len(sub_now)), sub_now, 1)[0]) if len(sub_now) > 1 else 0.0
            slope_prev = float(np.polyfit(np.arange(len(sub_prev)), sub_prev, 1)[0]) if len(sub_prev) > 1 else 0.0
        if t >= 36:
            lookback = 12
            window = df["close"].iloc[t-lookback:t+1]
            if df["close"].iloc[t] == float(window.min()) and slope_now > slope_prev:
                x = int(df["candle_index"].iloc[t])
                y = float(df["close"].iloc[t])
                signals.append({"index": x, "price": y})
                xs.append(x)
                ys.append(y)

    if viz:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df["candle_index"], df["close"], lw=1, label="Close Price", color="blue")
        ax.scatter(xs, ys, color="cyan", marker="v", s=100, zorder=6)
        ax.set_title("Price with Bottom Catcher (Brain 4)")
        ax.set_xlabel("Candles (Index)")
        ax.set_ylabel("Price")
        ax.grid(True)
        plt.show()

    return signals


def summarize(signals: List[Dict[str, float]], df: pd.DataFrame):
    """Compute valley-quality statistics for ▼ signals."""
    total = len(signals)
    indices = [int(s["index"]) for s in signals]
    avg_gap = int(np.mean(np.diff(indices))) if len(indices) > 1 else 0

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()

    # ATR(14) and its rolling median for volatility context
    prev_close = np.roll(closes, 1)
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)))
    tr[0] = highs[0] - lows[0]
    atr14 = pd.Series(tr).rolling(14).mean()
    atr_median = atr14.rolling(ATR_MED_WINDOW).median()

    sharp = 0
    slope_deltas: List[float] = []
    strong_slope = 0
    multi_confirm = 0
    high_vol = 0
    extrema_align = 0

    for idx in indices:
        if idx >= len(closes):
            continue

        # Valley sharpness via lower wick ratio
        rng = highs[idx] - lows[idx]
        if rng > 0:
            wick_ratio = (closes[idx] - lows[idx]) / rng
            if wick_ratio >= 0.5:
                sharp += 1

        # Slope delta between two 12-candle windows
        if idx >= 24:
            sub_now = closes[idx-12:idx]
            sub_prev = closes[idx-24:idx-12]
            slope_now = float(np.polyfit(np.arange(len(sub_now)), sub_now, 1)[0]) if len(sub_now) > 1 else 0.0
            slope_prev = float(np.polyfit(np.arange(len(sub_prev)), sub_prev, 1)[0]) if len(sub_prev) > 1 else 0.0
            delta = slope_now - slope_prev
        else:
            delta = 0.0
        slope_deltas.append(delta)
        if delta >= SLOPE_DELTA_THRESH:
            strong_slope += 1

        # Multi-window confirmation (12c & 48c minima)
        win12 = closes[max(0, idx-11):idx+1]
        win48 = closes[max(0, idx-47):idx+1]
        if win12.size and win48.size and win12[-1] == float(win12.min()) and win48[-1] == float(win48.min()):
            multi_confirm += 1

        # ATR context
        if not np.isnan(atr14.iloc[idx]) and not np.isnan(atr_median.iloc[idx]) and atr14.iloc[idx] > atr_median.iloc[idx]:
            high_vol += 1

        # Extrema alignment within ±5 candles
        start = max(0, idx - ALIGN_WINDOW)
        end = min(len(closes), idx + ALIGN_WINDOW + 1)
        region = closes[start:end]
        if region.size:
            min_pos = start + int(np.argmin(region))
            if abs(min_pos - idx) <= ALIGN_WINDOW:
                extrema_align += 1

    sharp_valley_pct = int(round(100 * sharp / total)) if total else 0
    slope_delta_avg = float(np.mean(slope_deltas)) if slope_deltas else 0.0
    slope_delta_strong_pct = int(round(100 * strong_slope / total)) if total else 0
    multiwin_confirm_pct = int(round(100 * multi_confirm / total)) if total else 0
    highvol_pct = int(round(100 * high_vol / total)) if total else 0
    extrema_align_pct = int(round(100 * extrema_align / total)) if total else 0

    print("[BRAIN][bottom_catcher][stats]")
    print(f"  Sharp valleys (wick ≥50%): {sharp_valley_pct}%")
    print(f"  Strong slope deltas (≥{SLOPE_DELTA_THRESH}): {slope_delta_strong_pct}%")
    print(f"  Multi-window confirmations: {multiwin_confirm_pct}%")
    print(f"  High-volatility context: {highvol_pct}%")
    print(f"  Extrema alignment (±{ALIGN_WINDOW}c): {extrema_align_pct}%")

    return {
        "count": total,
        "avg_gap": avg_gap,
        "sharp_valley_pct": sharp_valley_pct,
        "slope_delta_avg": round(slope_delta_avg, 6),
        "slope_delta_strong_pct": slope_delta_strong_pct,
        "multiwin_confirm_pct": multiwin_confirm_pct,
        "highvol_pct": highvol_pct,
        "extrema_align_pct": extrema_align_pct,
    }
