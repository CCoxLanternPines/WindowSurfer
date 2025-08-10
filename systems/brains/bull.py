from __future__ import annotations

import math

from ._utils import sma, slope, zscore


def _higher_highs(close, lookback, hh_min):
    if lookback <= 1 or len(close) < lookback:
        return False
    window = close[-lookback:]
    count = 0
    for i in range(1, len(window)):
        if window[i] > window[i - 1]:
            count += 1
    return count >= hh_min


def momo_long(candle_idx: int, series: dict, cfg: dict) -> bool:
    close = series["close"]
    M = cfg.get("M", 20)
    hh_lookback = cfg.get("hh_lookback", 5)
    hh_min = cfg.get("hh_min", 3)

    sma_M = sma(close, M)
    slope_M = slope(sma_M, M)
    z_M = zscore(close, M)

    if candle_idx >= len(close):
        return False
    sl = slope_M[candle_idx]
    ma = sma_M[candle_idx]
    z = z_M[candle_idx]
    if math.isnan(sl) or math.isnan(ma) or math.isnan(z):
        return False
    if sl <= 0.10 or z <= 1.5 or close[candle_idx] <= ma:
        return False
    start = max(0, candle_idx - hh_lookback + 1)
    return _higher_highs(close[start:candle_idx + 1], hh_lookback, hh_min)


def explain(candle_idx: int, series: dict, cfg: dict) -> dict:
    close = series["close"]
    M = cfg.get("M", 20)
    hh_lookback = cfg.get("hh_lookback", 5)
    hh_min = cfg.get("hh_min", 3)

    sma_M = sma(close, M)
    slope_M = slope(sma_M, M)
    z_M = zscore(close, M)

    if candle_idx >= len(close):
        return {"decision": False, "reasons": {}}
    sl = slope_M[candle_idx]
    ma = sma_M[candle_idx]
    z = z_M[candle_idx]
    slope_ok = not math.isnan(sl) and sl > 0.10
    z_ok = not math.isnan(z) and z > 1.5
    above_ma = not math.isnan(ma) and close[candle_idx] > ma
    start = max(0, candle_idx - hh_lookback + 1)
    hh = _higher_highs(close[start:candle_idx + 1], hh_lookback, hh_min)
    decision = slope_ok and z_ok and above_ma and hh
    reasons = {"slopeM_ok": slope_ok, "zM_ok": z_ok, "above_MA": above_ma, "higher_highs": hh}
    return {"decision": decision, "reasons": reasons}
