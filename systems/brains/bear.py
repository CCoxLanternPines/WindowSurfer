from __future__ import annotations

import math
from typing import Dict, Tuple

from ._utils import slope, zscore


# track slope hysteresis per series id
_slope_state: Dict[int, bool] = {}


def parked(candle_idx: int, series: dict, cfg: dict) -> Tuple[bool, dict]:
    """Determine if the bear brain should be parked.

    Returns a tuple of decision and debug info including trigger reason.
    """
    close = series["close"]
    L = cfg.get("L", 50)

    sl_arr = slope(close, L)
    z_arr = zscore(close, L)

    if candle_idx >= len(close):
        return False, {"slopeL": math.nan, "zL": math.nan, "reason": None}

    sl = sl_arr[candle_idx]
    z = z_arr[candle_idx]
    if math.isnan(sl) or math.isnan(z):
        return False, {"slopeL": sl, "zL": z, "reason": None}

    key = id(close)
    slope_triggered = _slope_state.get(key, False)
    reasons = []

    z_hit = z <= -1.0
    if z_hit:
        reasons.append("zscore")

    if sl <= -0.05 and not slope_triggered:
        slope_triggered = True
        reasons.append("slope")
    elif slope_triggered and sl >= 0.05:
        slope_triggered = False
    _slope_state[key] = slope_triggered

    if reasons:
        reason = "both" if len(reasons) == 2 else reasons[0]
        return True, {"slopeL": sl, "zL": z, "reason": reason}
    return False, {"slopeL": sl, "zL": z, "reason": None}


def explain(candle_idx: int, series: dict, cfg: dict) -> dict:
    decision, info = parked(candle_idx, series, cfg)
    return {"decision": decision, "reasons": info}

