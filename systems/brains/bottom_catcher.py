from __future__ import annotations

"""Brain 4: Bottom catcher with valley-quality statistics."""

from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..sim_engine import WINDOW_SIZE, WINDOW_STEP, multi_window_vote


ALIGN_WINDOW = 5  # ±5 candles for extrema alignment


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
    """Compute recovery-quality statistics for ▼ signals."""
    total = len(signals)
    indices = [int(s["index"]) for s in signals]
    avg_gap = int(np.mean(np.diff(indices))) if len(indices) > 1 else 0

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()

    recovery_mags: List[float] = []
    recovery_times: List[int] = []
    durable_lows = 0
    up_recovs: List[float] = []
    down_recovs: List[float] = []
    up_count = 0
    down_count = 0
    continuation = 0
    extrema_align = 0

    for idx in indices:
        if idx >= len(closes):
            continue

        price = closes[idx]

        # Recovery magnitude (max high within next 48 candles)
        future_highs = highs[idx + 1 : idx + 49]
        if future_highs.size:
            max_future = float(future_highs.max())
            rec_mag = (max_future / price - 1) * 100
        else:
            max_future = price
            rec_mag = 0.0
        recovery_mags.append(rec_mag)

        # Time-to-Recovery: candles to +2% close within 48 candles
        target = price * 1.02
        future_closes = closes[idx + 1 : idx + 49]
        for j, c in enumerate(future_closes, start=1):
            if c >= target:
                recovery_times.append(j)
                break

        # Follow-through Probability: low not revisited within 24 candles
        future_lows = lows[idx + 1 : idx + 25]
        if future_lows.size and future_lows.min() > lows[idx]:
            durable_lows += 1

        # Trend context via multi-window vote
        decision, _, _ = multi_window_vote(df, idx, window_sizes=[8, 12, 24, 48])
        if decision == 1:
            up_recovs.append(rec_mag)
            up_count += 1
        elif decision == -1:
            down_recovs.append(rec_mag)
            down_count += 1

        # Continuation Lift: exceed prior swing high within 48 candles
        prev_high = highs[max(0, idx - 48) : idx].max() if idx > 0 else price
        if future_highs.size and future_highs.max() > prev_high:
            continuation += 1

        # Extrema alignment within ±5 candles
        start = max(0, idx - ALIGN_WINDOW)
        end = min(len(closes), idx + ALIGN_WINDOW + 1)
        region = closes[start:end]
        if region.size:
            min_pos = start + int(np.argmin(region))
            if abs(min_pos - idx) <= ALIGN_WINDOW:
                extrema_align += 1

    recovery_mag = float(np.mean(recovery_mags)) if recovery_mags else 0.0
    time_to_recovery = float(np.mean(recovery_times)) if recovery_times else 0.0
    durable_low_pct = int(round(100 * durable_lows / total)) if total else 0
    trend_bias_up = float(np.mean(up_recovs)) if up_recovs else 0.0
    trend_bias_down = float(np.mean(down_recovs)) if down_recovs else 0.0
    continuation_lift_pct = int(round(100 * continuation / total)) if total else 0
    extrema_align_pct = int(round(100 * extrema_align / total)) if total else 0

    ud_total = up_count + down_count
    if ud_total:
        pct = int(round(100 * max(up_count, down_count) / ud_total))
        direction = "uptrend" if up_count >= down_count else "downtrend"
    else:
        pct = 0
        direction = "neutral"
    slope_bias = f"{direction} {pct}%"

    print("[BRAIN][bottom_catcher][stats]")
    print(f"  Avg recovery magnitude: {recovery_mag:+.1f}%")
    print(f"  Time-to-recovery (2%): {time_to_recovery:.0f}c")
    print(f"  Durable lows: {durable_low_pct}%")
    print(
        f"  Trend bias: uptrend {trend_bias_up:.1f}% avg bounce, downtrend {trend_bias_down:.1f}% avg bounce"
    )
    print(f"  Continuation lift: {continuation_lift_pct}%")
    print(f"  Extrema alignment: {extrema_align_pct}%")

    return {
        "count": total,
        "avg_gap": avg_gap,
        "slope_bias": slope_bias,
        "recovery_mag": round(recovery_mag, 3),
        "time_to_recovery": round(time_to_recovery, 3),
        "durable_low_pct": durable_low_pct,
        "trend_bias_up": round(trend_bias_up, 3),
        "trend_bias_down": round(trend_bias_down, 3),
        "continuation_lift_pct": continuation_lift_pct,
        "extrema_align_pct": extrema_align_pct,
    }
