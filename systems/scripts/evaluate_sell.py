from __future__ import annotations

"""Sell evaluation based on predictive pressures."""

from typing import Any, Dict, List

from systems.scripts.evaluate_buy import (
    classify_slope,
    compute_window_features,
)
from systems.utils.addlog import addlog
from systems.utils.settings_loader import load_coin_settings


def evaluate_sell(
    ctx: Dict[str, Any],
    t: int,
    series,
    *,
    cfg: Dict[str, Any] | None = None,
    open_notes: List[Dict[str, Any]],
    runtime_state: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Return a list of notes to sell on this candle."""

    runtime_state = runtime_state or {}
    if cfg is None:
        market = runtime_state.get("kraken_name") or runtime_state.get("market", "")
        cfg = load_coin_settings(market)

    window_name = "strategy"
    strategy = cfg
    window_size = int(strategy["window_size"])
    window_step = int(strategy["window_step"])

    verbose = runtime_state.get("verbose", 0)

    pressures = runtime_state.setdefault("pressures", {"buy": {}, "sell": {}})
    sell_p = pressures["sell"].get(window_name, 0.0)
    buy_p = pressures["buy"].get(window_name, 0.0)
    max_p = strategy["max_pressure"]
    results: List[Dict[str, Any]] = []
    window_notes = open_notes
    n_notes = len(window_notes)
    candle = series.iloc[t]
    price = float(candle["close"])

    features = runtime_state.get("last_features", {}).get(window_name)
    if features is None:
        features = compute_window_features(series, t, strategy)
    slope = features.get("slope", 0.0)
    def roi_now(note: Dict[str, Any]) -> float:
        buy = note.get("entry_price", 0.0)
        return (price - buy) / buy if buy else 0.0

    window_notes.sort(key=roi_now, reverse=True)

    sell_trigger = strategy["sell_trigger"]

    if sell_p >= max_p and window_notes:
        all_count = strategy.get("all_sell_count")
        if all_count is None:
            all_count = n_notes
        k = min(all_count, n_notes)
        if k <= 0:
            return []
        results = window_notes[:k]
        for n in results:
            n["sell_mode"] = "all"
        pressures["sell"][window_name] = 0.0
        addlog(
            f"[SELL][{window_name} {window_size}] mode=all count={k}/{n_notes}",
            verbose_int=1,
            verbose_state=verbose,
        )
        return results

    slope_cls = classify_slope(slope, strategy)
    if slope_cls == 0 and sell_p >= sell_trigger:
        if window_notes:
            flat_percent = strategy["flat_sell_percent"]
            k = int(round(n_notes * flat_percent))
            if k <= 0:
                return []
            results = window_notes[:k]
            for n in results:
                n["sell_mode"] = "flat"
            pressures["sell"][window_name] = 0.0
            addlog(
                f"[SELL][{window_name} {window_size}] mode=flat count={k}/{n_notes} "
                f"flat_percent={flat_percent:.2%}",
                verbose_int=1,
                verbose_state=verbose,
            )
            return results
        addlog(
            f"[HOLD][{window_name} {window_size}] flat sell condition met but no notes",
            verbose_int=2,
            verbose_state=verbose,
        )
        return []

    addlog(
        f"[HOLD][{window_name} {window_size}] need={sell_trigger:.2f}, have={sell_p:.2f}, buy_p={buy_p:.2f}",
        verbose_int=2,
        verbose_state=verbose,
    )

    return []

