from __future__ import annotations

"""Stepwise forecast helper for slope projections."""

import numpy as np
import pandas as pd


def forecast_stepwise(
    df: pd.DataFrame, bottom_window: int, k1: float = 0.2, k2: float = 0.3
) -> list[float]:
    """Predict a per-tick forecast line.

    Starting from the first candle's close, iteratively step forward using the
    slope of the recent ``bottom_window`` closes plus volume and breakout bias.
    """

    forecast = [np.nan] * len(df)

    # Initialize at the first candle's actual close
    forecast[0] = df["close"].iloc[0]

    for i in range(1, len(df)):
        start = max(0, i - bottom_window)
        y = df["close"].iloc[start:i].values
        x = np.arange(len(y))

        if len(y) > 1:
            m, b = np.polyfit(x, y, 1)

            # Base slope prediction from previous forecast value
            step_pred = forecast[i - 1] + m

            # Volume bias
            recent_vol = df["volume"].iloc[start:i]
            volume_change = (recent_vol.iloc[-1] - recent_vol.iloc[0]) / max(
                1e-9, recent_vol.iloc[0]
            )

            # Breakout bias
            last_close = df["close"].iloc[i - 1]
            last_fit = m * (len(y) - 1) + b
            breakout_dist = last_close - last_fit

            adjustment = k1 * volume_change - k2 * breakout_dist

            forecast[i] = step_pred + adjustment
        else:
            forecast[i] = forecast[i - 1]

    return forecast


__all__ = ["forecast_stepwise"]

