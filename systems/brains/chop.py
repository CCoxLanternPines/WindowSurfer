from __future__ import annotations

import math

from ._utils import sma, zscore, slope


def edge_long(candle_idx: int, series: dict, cfg: dict) -> bool:
    close = series["close"]
    S = cfg.get("S", 20)
    z_buy = cfg.get("z_buy", -1.0)
    slope_w = cfg.get("slope_w", 5)

    z_S = zscore(close, S)
    sma_slope = slope(sma(close, slope_w), slope_w)

    if candle_idx >= len(close):
        return False
    z = z_S[candle_idx]
    sl = sma_slope[candle_idx]
    if math.isnan(z) or math.isnan(sl):
        return False
    return z <= z_buy and sl >= 0
