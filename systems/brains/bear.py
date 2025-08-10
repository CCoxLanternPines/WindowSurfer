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


def explain(candle_idx: int, series: dict, cfg: dict) -> dict:
    close = series["close"]
    high = series["high"]
    low = series["low"]
    L = cfg.get("L", 50)
    z_off = cfg.get("z_off", -1.0)
    atr_cool = cfg.get("atr_cool", 0.02)

    sma_L = sma(close, L)
    z_L = zscore(close, L)
    slope_L = slope(sma_L, L)
    atr_L = atr(high, low, close, L)

    if candle_idx >= len(close):
        return {"decision": False, "reasons": {}}
    sl = slope_L[candle_idx]
    z = z_L[candle_idx]
    atr_rel = atr_L[candle_idx] / close[candle_idx] if close[candle_idx] else math.nan
    atr_ok = not math.isnan(atr_rel) and atr_rel < atr_cool
    if math.isnan(sl) or math.isnan(z):
        decision = False
    else:
        decision = sl < 0 or z < z_off
    reasons = {"slopeL": sl, "zL": z, "atr_cool_ok": atr_ok}
    return {"decision": decision, "reasons": reasons}
