from __future__ import annotations

import math
from typing import Dict

from ._utils import slope, zscore, sma


# track hysteresis per series id
_slope_state: Dict[int, bool] = {}


def parked_explain(i: int, series: dict, cfg: dict):
    """Return bear decision and debug info without hysteresis state."""

    L = int(cfg.get("L", 50))
    M = int(cfg.get("M", 20))
    close = series["close"]

    zL = zscore(close, L)[i]
    slopeL = slope(close, L)[i]
    slopeM = slope(close, M)[i]
    smaM = sma(close, M)[i]

    cond_z = zL <= -1.5
    cond_trend = slopeM <= 0
    cond_ma = close[i] < smaM

    decision = bool(cond_z and cond_trend and cond_ma)
    reason = "zscore+trend+ma" if decision else None
    return decision, {
        "zL": zL,
        "slopeL": slopeL,
        "slopeM": slopeM,
        "above_M": int(close[i] >= smaM),
        "reason": reason,
    }


def parked(i: int, series: dict, cfg: dict) -> bool:
    """Return True when bear brain is parked using hysteresis."""

    dec, info = parked_explain(i, series, cfg)
    key = id(series["close"])
    if dec:
        _slope_state[key] = True
        return True
    if _slope_state.get(key, False):
        if info["slopeL"] >= 0.05 and abs(info["zL"]) <= 0.5:
            _slope_state[key] = False
            return False
        return True
    return False


def explain(candle_idx: int, series: dict, cfg: dict) -> dict:
    decision, reasons = parked_explain(candle_idx, series, cfg)
    return {"decision": decision, "reasons": reasons}

