from __future__ import annotations

"""Shared math constants and helpers for brains and simulation."""

import numpy as np
import pandas as pd

WINDOW_SIZE = 100
WINDOW_STEP = 1
CLUSTER_WINDOW = 12
BASE_SIZE = 40
SCALE_POWER = 1.3


def multi_window_vote(df: pd.DataFrame, t: int, window_sizes, slope_thresh: float = 0.001, range_thresh: float = 0.05):
    """Return (-1,0,1) decision with confidence using multi-window slope direction."""
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
