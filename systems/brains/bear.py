from __future__ import annotations

import math

from ._utils import sma, zscore, slope, atr


def parked(candle_idx: int, series: dict, cfg: dict) -> bool:
    close = series["close"]
    L = cfg.get("L", 50)
    z_off = cfg.get("z_off", -1.0)

    sma_L = sma(close, L)
    z_L = zscore(close, L)
    slope_L = slope(sma_L, L)

    if candle_idx >= len(close):
        return False
    sl = slope_L[candle_idx]
    z = z_L[candle_idx]
    if math.isnan(sl) or math.isnan(z):
        return False
    return sl < 0 or z < z_off
