from __future__ import annotations

import argparse
import re
from datetime import timedelta
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Parameters ---
WINDOW_SIZE = 24       # candles per box
WINDOW_STEP = 2        # step between boxes
CLUSTER_WINDOW = 10    # lookback to count exhaustion density
BASE_SIZE = 10         # base dot size
SCALE_POWER = 2        # exhaustion growth

# --- Helpers ---
def parse_timeframe(tf: str) -> timedelta | None:
    m = re.match(r"(\d+)([dhmw])", tf)
    if not m:
        return None
    n, u = int(m.group(1)), m.group(2)
    return {
        'h': timedelta(hours=n),
        'd': timedelta(days=n),
        'w': timedelta(weeks=n),
        'm': timedelta(days=30*n)  # rough
    }[u]

def multi_window_vote(df, t, window_sizes, slope_thresh=0.001, range_thresh=0.05):
    votes, strengths = [], []
    for W in window_sizes:
        if t - W < 0:
            continue
        sub = df.iloc[t-W:t]
        closes = sub['close'].values
        x = np.arange(len(closes))
        slope = float(np.polyfit(x, closes, 1)[0]) if len(closes) > 1 else 0.0
        rng = float(sub['close'].max() - sub['close'].min())
        if abs(slope) < slope_thresh or rng < range_thresh:
            continue
        direction = 1 if slope > 0 else -1
        votes.append(direction)
        strengths.append(abs(slope) * rng)
    score = sum(votes)
    confidence = (sum(strengths) / max(1, len(strengths))) if strengths else 0.0
    if score >= 2:   # momentum up
        return 1, confidence, score
    if score <= -2:  # momentum down
        return -1, confidence, score
    return 0, confidence, score

# --- Main ---
def run_simulation(*, timeframe: str = "1m", viz: bool = True) -> None:
    df = pd.read_csv("data/sim/SOLUSD_1h.csv")
    if timeframe:
        delta = parse_timeframe(timeframe)
        if delta:
            cutoff = (pd.Timestamp.utcnow().tz_localize(None) - delta).timestamp()
            df = df[df['timestamp'] >= cutoff]
    df = df.reset_index(drop=True)
    df['candle_index'] = range(len(df))

    if viz:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(df['candle_index'], df['close'], lw=1, label='Close Price', color='blue')

    # State
    recent_buys = deque(maxlen=CLUSTER_WINDOW)
    recent_sells = deque(maxlen=CLUSTER_WINDOW)
    exhaustion_handles: list = []
    reversal_handles: list = []
    inflection_handles: list = []   # key 3
    bottom_handles: list = []       # key 4
    top5_handles: list = []         # key 5 (Divergence)
    top6_handles: list = []         # key 6 (Rolling peak)
    top7_handles: list = []         # key 7 (Compression->Expansion)
    top8_handles: list = []         # key 8 (Meta-filter: reversal + divergence)

    last_exhaustion_decision: int | None = None

    # Iterate
    for t in range(WINDOW_SIZE - 1, len(df), WINDOW_STEP):
        candle = df.iloc[t]
        decision, confidence, score = multi_window_vote(df, t, window_sizes=[8, 12, 24, 48])

        # --- Exhaustion (Key 1 baseline, but toggleable) ---
        if decision == 1:  # SELL exhaustion
            recent_buys.append(t)
            cluster_strength = sum(1 for idx in recent_buys if t - idx <= CLUSTER_WINDOW)
            size = BASE_SIZE * (cluster_strength ** SCALE_POWER)
            if viz:
                h = ax1.scatter(candle['candle_index'], candle['close'], color='red', s=size, zorder=6)
                exhaustion_handles.append(h)
        elif decision == -1:  # BUY exhaustion
            recent_sells.append(t)
            cluster_strength = sum(1 for idx in recent_sells if t - idx <= CLUSTER_WINDOW)
            size = BASE_SIZE * (cluster_strength ** SCALE_POWER)
            if viz:
                h = ax1.scatter(candle['candle_index'], candle['close'], color='green', s=size, zorder=6)
                exhaustion_handles.append(h)

        # --- Reversals (Key 2: yellow) ---
        if last_exhaustion_decision is not None and decision != 0 and decision != last_exhaustion_decision:
            if viz:
                h2 = ax1.scatter(candle['candle_index'], candle['close'],
                                  color='yellow', s=120, zorder=7, edgecolor='black')
                reversal_handles.append(h2)
        if decision != 0:
            last_exhaustion_decision = decision

        # Precompute slopes for 24c windows (used by several keys)
        slope_now = 0.0
        slope_prev = 0.0
        if t >= 48:
            sub_now = df['close'].iloc[t-24:t]
            sub_prev = df['close'].iloc[t-48:t-24]
            slope_now = float(np.polyfit(np.arange(len(sub_now)), sub_now, 1)[0]) if len(sub_now) > 1 else 0.0
            slope_prev = float(np.polyfit(np.arange(len(sub_prev)), sub_prev, 1)[0]) if len(sub_prev) > 1 else 0.0

        # --- Key 3: Momentum Inflection (from prior 6) ---
        # Mark when slope weakens sharply or flips sign
        if t >= 48:
            if (slope_prev > 0 and slope_now < 0) or (slope_prev < 0 and slope_now > 0) or (abs(slope_now) < 0.6 * abs(slope_prev)):
                if viz:
                    h3 = ax1.scatter(candle['candle_index'], candle['close'], color='orange', marker='^', s=80, zorder=5)
                    inflection_handles.append(h3)

        # --- Key 4: Bottom Catcher (from prior 7) ---
        # Local minimum with improving slope
        if t >= 36:
            lookback = 12
            window = df['close'].iloc[t-lookback:t+1]
            if candle['close'] == window.min() and slope_now > slope_prev:
                if viz:
                    h4 = ax1.scatter(candle['candle_index'], candle['close'], color='cyan', marker='v', s=100, zorder=6)
                    bottom_handles.append(h4)

        # --- Key 5: Divergence (Top catcher - confirmation) ---
        if t >= 48:
            price_now = df['close'].iloc[t-1]
            price_prev = df['close'].iloc[t-25]
            # Bearish divergence: higher price, weaker momentum
            if price_now > price_prev and slope_now < slope_prev:
                if viz:
                    h5 = ax1.scatter(candle['candle_index'], candle['close'], color='orange', marker='s', s=110, zorder=6)
                    top5_handles.append(h5)

        # --- Key 6: Rolling Peak Detection (Top catcher - swing high) ---
        if t >= 12:
            lookback = 12
            win = df['close'].iloc[t-lookback:t+1]
            if candle['close'] == win.max():
                if viz:
                    h6 = ax1.scatter(candle['candle_index'], candle['close'], color='red', marker='*', s=140, zorder=6)
                    top6_handles.append(h6)

        # --- Key 7: Compression -> Expansion at highs (Top catcher - predictive) ---
        if t >= 18:
            rng_window = 12
            sub_rng = df['close'].iloc[t-rng_window:t]
            rng = float(sub_rng.max() - sub_rng.min())
            # Squeeze threshold (looser so it actually shows)
            if rng < 0.02 * float(sub_rng.mean()) and decision == 1:
                if viz:
                    h7 = ax1.scatter(candle['candle_index'], candle['close'], color='purple', marker='^', s=110, zorder=6)
                    top7_handles.append(h7)

        # --- Key 8: Meta-Filter (high conviction overlap: reversal + divergence) ---
        if viz and len(reversal_handles) > 0 and len(top5_handles) > 0:
            # If most recent reversal is near most recent divergence -> star
            x_rev, y_rev = reversal_handles[-1].get_offsets()[0]
            x_div, y_div = top5_handles[-1].get_offsets()[0]
            if abs(x_rev - x_div) <= 2:
                h8 = ax1.scatter(x_rev, y_rev, color='magenta', marker='P', s=160, zorder=7)
                top8_handles.append(h8)

    if viz:
        ax1.set_title('Price with Exhaustion + Predictors (Keys 1–8)')
        ax1.set_xlabel('Candles (Index)')
        ax1.set_ylabel('Price')
        ax1.grid(True)

        state = {
            'exhaustion_on': True,
            'reversal_on': True,
            'inflection_on': True,
            'bottom_on': True,
            'top5_on': True,
            'top6_on': True,
            'top7_on': True,
            'top8_on': True,
        }

        def on_key(event):
            if event.key == '1':
                state['exhaustion_on'] = not state['exhaustion_on']
                for h in exhaustion_handles:
                    h.set_visible(state['exhaustion_on'])
                print(f"[TOGGLE] Exhaustion {'ON' if state['exhaustion_on'] else 'OFF'}"); plt.draw()
            elif event.key == '2':
                state['reversal_on'] = not state['reversal_on']
                for h in reversal_handles:
                    h.set_visible(state['reversal_on'])
                print(f"[TOGGLE] Reversals {'ON' if state['reversal_on'] else 'OFF'}"); plt.draw()
            elif event.key == '3':
                state['inflection_on'] = not state['inflection_on']
                for h in inflection_handles:
                    h.set_visible(state['inflection_on'])
                print(f"[TOGGLE] Inflection {'ON' if state['inflection_on'] else 'OFF'}"); plt.draw()
            elif event.key == '4':
                state['bottom_on'] = not state['bottom_on']
                for h in bottom_handles:
                    h.set_visible(state['bottom_on'])
                print(f"[TOGGLE] Bottom Catcher {'ON' if state['bottom_on'] else 'OFF'}"); plt.draw()
            elif event.key == '5':
                state['top5_on'] = not state['top5_on']
                for h in top5_handles:
                    h.set_visible(state['top5_on'])
                print(f"[TOGGLE] Top 5 (Divergence) {'ON' if state['top5_on'] else 'OFF'}"); plt.draw()
            elif event.key == '6':
                state['top6_on'] = not state['top6_on']
                for h in top6_handles:
                    h.set_visible(state['top6_on'])
                print(f"[TOGGLE] Top 6 (Rolling Peak) {'ON' if state['top6_on'] else 'OFF'}"); plt.draw()
            elif event.key == '7':
                state['top7_on'] = not state['top7_on']
                for h in top7_handles:
                    h.set_visible(state['top7_on'])
                print(f"[TOGGLE] Top 7 (Compression) {'ON' if state['top7_on'] else 'OFF'}"); plt.draw()
            elif event.key == '8':
                state['top8_on'] = not state['top8_on']
                for h in top8_handles:
                    h.set_visible(state['top8_on'])
                print(f"[TOGGLE] Top 8 (Meta) {'ON' if state['top8_on'] else 'OFF'}"); plt.draw()

        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show()

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--time', type=str, default='1m')
    p.add_argument('--viz', action='store_true')
    args = p.parse_args()
    run_simulation(timeframe=args.time, viz=args.viz)

if __name__ == '__main__':
    main()
