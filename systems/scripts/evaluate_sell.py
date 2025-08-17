from __future__ import annotations

"""Sell evaluation based on predictive pressures."""

from math import ceil
from typing import Any, Dict, List

from systems.scripts.trend_predict import (
    compute_window_features,
    rule_predict,
    update_pressures,
    classify_slope,
)
from systems.utils.addlog import addlog


def evaluate_sell(
    ctx: Dict[str, Any],
    t: int,
    series,
    *,
    window_name: str,
    cfg: Dict[str, Any],
    open_notes: List[Dict[str, Any]],
    runtime_state: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Return a list of notes to sell in ``window_name`` on this candle."""

    runtime_state = runtime_state or {}
    strategy = runtime_state.get("strategy", {})
    window_size = int(strategy.get("window_size") or cfg.get("window_size", 0))
    step = int(strategy.get("window_step", 1))
    start = t + 1 - window_size
    if start < 0 or start % step != 0:
        return []

    verbose = runtime_state.get("verbose", 0)

    features = compute_window_features(series, start, window_size)
    last = runtime_state.setdefault("last_features", {}).get(window_name)
    if last is not None:
        pred = rule_predict(last, strategy)
        slope_cls = classify_slope(last.get("slope", 0.0), strategy.get("flat_band_deg", 10.0))
        update_pressures(runtime_state, window_name, pred, slope_cls, strategy)
    runtime_state["last_features"][window_name] = features

    pressures = runtime_state.setdefault("pressures", {"buy": {}, "sell": {}})
    sell_p = pressures["sell"].get(window_name, 0.0)
    max_p = strategy.get("max_pressure", 1.0)
    results: List[Dict[str, Any]] = []
    window_notes = [n for n in open_notes if n.get("window_name") == window_name]
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
        pressures["sell"][window_name] = 0.0
        addlog(
            f"[SELL][{window_name} {window_size}] mode=normal count={k}/{n_notes} pressure={sell_p:.1f}/{max_p:.1f}",
            verbose_int=1,
            verbose_state=verbose,
        )
        return results

    slope_cls = classify_slope(last.get("slope", 0.0) if last else 0.0, strategy.get("flat_band_deg", 10.0))
    flat_trigger = strategy.get("sell_trigger", 0.0) * strategy.get("flat_sell_threshold", 1.0)
    if (
        not results
        and slope_cls == 0
        and sell_p >= flat_trigger
        and window_notes
    ):
        k = max(1, ceil(strategy.get("flat_sell_fraction", 0.0) * n_notes))
        results = window_notes[:k]
        pressures["sell"][window_name] = 0.0
        addlog(
            f"[SELL][{window_name} {window_size}] mode=flat count={k}/{n_notes} pressure={sell_p:.1f}/{max_p:.1f}",
            verbose_int=1,
            verbose_state=verbose,
        )
        return results

    return []
