from __future__ import annotations

"""Sell evaluation based on predictive pressures."""

from math import ceil
from typing import Any, Dict, List

from systems.scripts.evaluate_buy import (
    classify_slope,
    compute_window_features,
)
from systems.utils.addlog import addlog


def evaluate_sell(
    ctx: Dict[str, Any],
    t: int,
    series,
    *,
    cfg: Dict[str, Any],
    open_notes: List[Dict[str, Any]],
    runtime_state: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Return a list of notes to sell on this candle."""

    runtime_state = runtime_state or {}
    window_name = "strategy"
    strategy = cfg or runtime_state.get("strategy", {})
    window_size = int(strategy.get("window_size", 0))

    verbose = runtime_state.get("verbose", 0)

    pressures = runtime_state.setdefault("pressures", {"buy": {}, "sell": {}})
    sell_p = pressures["sell"].get(window_name, 0.0)
    max_p = strategy.get("max_pressure", 1.0)
    results: List[Dict[str, Any]] = []
    window_notes = open_notes
    n_notes = len(window_notes)
    candle = series.iloc[t]
    price = float(candle["close"])

    def roi_now(note: Dict[str, Any]) -> float:
        buy = note.get("entry_price", 0.0)
        return (price - buy) / buy if buy else 0.0

    window_notes.sort(key=roi_now, reverse=True)

    if sell_p >= strategy.get("sell_trigger", 0.0) and window_notes:
        sell_frac = sell_p / max_p if max_p else 0.0
        k = max(1, ceil(sell_frac * n_notes))
        results = window_notes[:k]
        for n in results:
            n["sell_mode"] = "normal"
        pressures["sell"][window_name] = 0.0
        addlog(
            f"[SELL][{window_name} {window_size}] mode=normal count={k}/{n_notes} pressure={sell_p:.1f}/{max_p:.1f}",
            verbose_int=1,
            verbose_state=verbose,
        )
        return results

    # Recompute slope classification from the latest window features
    features = runtime_state.get("last_features", {}).get(window_name)
    if features is None:
        features = compute_window_features(series, t, window_size)
    slope_cls = classify_slope(
        features.get("slope", 0.0), strategy.get("flat_band_deg", 10.0)
    )
    flat_trigger = strategy.get("sell_trigger", 0.0) * strategy.get(
        "flat_sell_threshold", 1.0
    )
    if slope_cls == 0 and sell_p >= flat_trigger and window_notes:
        k = max(1, ceil(strategy.get("flat_sell_fraction", 0.0) * n_notes))
        results = window_notes[:k]
        for n in results:
            n["sell_mode"] = "flat"
        pressures["sell"][window_name] = 0.0
        addlog(
            f"[SELL][{window_name} {window_size}] mode=flat count={k}/{n_notes} pressure={sell_p:.1f}/{max_p:.1f}",
            verbose_int=1,
            verbose_state=verbose,
        )
        return results


    return []
