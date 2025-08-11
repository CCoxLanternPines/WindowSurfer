from __future__ import annotations

"""Helpers for window-based price normalisation."""

from typing import Tuple
import pandas as pd


def _parse_window_size(window_size: str) -> int:
    """Return number of candles corresponding to ``window_size``.

    Supported suffixes:
    ``h`` for hours, ``d`` for days, ``w`` for weeks. The dataset is assumed to
    use one-hour candles, so values are converted accordingly.
    """
    if not window_size:
        return 0
    try:
        value = int(window_size[:-1])
    except ValueError:  # pragma: no cover - invalid config
        value = 0
    unit = window_size[-1].lower()
    if unit == "h":
        factor = 1
    elif unit == "d":
        factor = 24
    elif unit == "w":
        factor = 24 * 7
    else:  # pragma: no cover - unsupported unit
        factor = 0
    return value * factor


def get_window_bounds(series: pd.DataFrame, t: int, window_size: str) -> Tuple[float, float]:
    """Return the ``(low, high)`` price bounds for a trailing window.

    The window spans ``window_size`` ending at index ``t`` (inclusive). If the
    window extends before the start of ``series`` the available range is used.
    """
    span = _parse_window_size(window_size)
    start = max(0, t - span + 1)
    window = series.iloc[start : t + 1]
    win_low = float(window["low"].min()) if "low" in window else float(window["close"].min())
    win_high = float(window["high"].max()) if "high" in window else float(window["close"].max())
    return win_low, win_high


def get_window_position(price: float, win_low: float, win_high: float) -> float:
    """Normalise ``price`` within the window bounds to a 0..1 range."""
    if win_high == win_low:
        return 0.5
    return max(0.0, min(1.0, (price - win_low) / (win_high - win_low)))
