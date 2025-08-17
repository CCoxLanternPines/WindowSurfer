from __future__ import annotations

"""Mini-bot compatible sell evaluation."""

from typing import Any, Dict, List

from .evaluate_buy import classify_slope


def evaluate_sell(
    t: int,
    series,
    *,
    cfg: Dict[str, float],
    open_notes: List[Dict[str, float]],
    state: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Return proportional sell instructions for this tick."""

    sell_p = state.get("sell_pressure", 0.0)
    max_p = float(cfg.get("max_pressure", 7.0))
    results: List[Dict[str, Any]] = []

    total_size = sum(n.get("amount", 0.0) for n in open_notes)
    if sell_p >= cfg.get("sell_trigger", 0.0) and total_size > 0:
        sell_frac = sell_p / max_p if max_p else 0.0
        for n in open_notes:
            amt = sell_frac * n.get("amount", 0.0)
            if amt > 0:
                results.append({"note": n, "sell_amount": amt, "sell_mode": "normal"})
        state["sell_pressure"] = 0.0
        return results

    features = state.get("last_features")
    slope_cls = (
        classify_slope(features.get("slope", 0.0), cfg.get("flat_band_deg", 10.0))
        if features
        else None
    )
    flat_trigger = cfg.get("sell_trigger", 0.0) * cfg.get("flat_sell_threshold", 1.0)
    if slope_cls == 0 and sell_p >= flat_trigger and total_size > 0:
        frac = cfg.get("flat_sell_fraction", 0.0)
        for n in open_notes:
            amt = frac * n.get("amount", 0.0)
            if amt > 0:
                results.append({"note": n, "sell_amount": amt, "sell_mode": "flat"})
        state["sell_pressure"] = 0.0
        return results

    return []

