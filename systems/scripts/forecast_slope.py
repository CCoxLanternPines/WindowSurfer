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
    anchor_val: float | None = None,
) -> np.ndarray:
    """Project slope forward ``bottom_window`` steps with bias adjustment.

    If ``anchor_val`` is provided, the forecast is shifted so its first value
    starts from ``anchor_val`` instead of the raw intercept, ensuring continuity
    across segments.
    """

    x_future = np.arange(bottom_window)

    # Base projection
    y_future = m * x_future + b

    # Bias: volume + breakout distance
    recent_vol = df["volume"].iloc[start_idx:end_idx]
    volume_change = (
        recent_vol.iloc[-1] - recent_vol.iloc[0]
    ) / max(1e-9, recent_vol.iloc[0])
    last_close = df["close"].iloc[end_idx - 1]
    last_fit = m * (len(recent_vol) - 1) + b
    breakout_dist = last_close - last_fit

    k1, k2 = 0.2, 0.3
    adj = k1 * volume_change - k2 * breakout_dist
    y_future = y_future + adj

    # Anchor continuity
    if anchor_val is not None:
        offset = anchor_val - y_future[0]
        y_future = y_future + offset

    return y_future
