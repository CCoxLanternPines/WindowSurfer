from __future__ import annotations

"""Forecast helper for projecting slope segments."""

import numpy as np
import pandas as pd


def forecast_slope_segment(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    m: float,
    b: float,
    bottom_window: int,
) -> np.ndarray:
    """Project slope line forward ``bottom_window`` steps with bias.

    The projection starts from the current slope fit (``m`` and ``b``), then
    applies a simple bias based on recent volume change and breakout distance.
    """
    # Base projection
    x_future = np.arange(bottom_window)
    y_future = m * x_future + b

    # Feature signals
    recent_vol = df["volume"].iloc[start_idx:end_idx]
    volume_change = (
        recent_vol.iloc[-1] - recent_vol.iloc[0]
    ) / max(1e-9, recent_vol.iloc[0])
    last_close = df["close"].iloc[end_idx - 1]
    last_fit = m * (len(recent_vol) - 1) + b
    breakout_dist = last_close - last_fit

    # Bias adjustment (tunable weights)
    k1, k2 = 0.2, 0.3
    adj = k1 * volume_change - k2 * breakout_dist

    y_future = y_future + adj
    return y_future
