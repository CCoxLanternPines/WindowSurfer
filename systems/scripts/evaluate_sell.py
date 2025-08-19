from __future__ import annotations

"""Sell evaluation based on predictive pressures."""

from typing import Any, Dict, List

from systems.scripts.evaluate_buy import (
    classify_slope,
    compute_window_features,
)
from systems.utils.addlog import addlog, send_telegram_message
from systems.utils.telegram_utils import describe_trend, format_window_status


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
    buy_p = pressures["buy"].get(window_name, 0.0)
    max_p = strategy.get("max_pressure", 1.0)
    results: List[Dict[str, Any]] = []
    window_notes = open_notes
    n_notes = len(window_notes)
    candle = series.iloc[t]
    price = float(candle["close"])

    features = runtime_state.get("last_features", {}).get(window_name)
    if features is None:
        features = compute_window_features(series, t, window_size)
    slope = features.get("slope", 0.0)
    volatility = features.get("volatility", 0.0)
    trend = describe_trend(slope)

    def roi_now(note: Dict[str, Any]) -> float:
        buy = note.get("entry_price", 0.0)
        return (price - buy) / buy if buy else 0.0

    window_notes.sort(key=roi_now, reverse=True)

    symbol = runtime_state.get("symbol", "")
    window_label = f"{window_size}h"
    current_pos = buy_p - sell_p
    sell_trigger = strategy.get("sell_trigger", 0.0)
    buy_trigger = strategy.get("buy_trigger", 0.0)

    if sell_p >= max_p and window_notes:
        all_count = strategy.get("all_sell_count", n_notes)
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
        total_usd = sum(n.get("entry_amount", 0.0) * price for n in results)
        action = f"SELL ${total_usd:.2f} ({k}/{n_notes} notes)"
        note = (
            f"id={results[0]['id']} price={price:.2f} count={k}/{n_notes}"
            if results
            else None
        )
        msg = format_window_status(
            symbol,
            window_label,
            trend,
            slope,
            volatility,
            buy_trigger,
            current_pos,
            sell_trigger,
            n_notes,
            "SELL (confident exit)",
            action,
            note,
        )
        send_telegram_message(msg)
        return results

    slope_cls = classify_slope(
        slope, strategy.get("flat_band_deg", 10.0)
    )
    if slope_cls == 0 and sell_p >= sell_trigger:
        if window_notes:
            flat_percent = strategy.get("flat_sell_percent", 0.0)
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
            total_usd = sum(n.get("entry_amount", 0.0) * price for n in results)
            action = f"SELL ${total_usd:.2f} ({k}/{n_notes} notes)"
            note = (
                f"id={results[0]['id']} price={price:.2f} count={k}/{n_notes}"
                if results
                else None
            )
            msg = format_window_status(
                symbol,
                window_label,
                trend,
                slope,
                volatility,
                buy_trigger,
                current_pos,
                sell_trigger,
                n_notes,
                "SELL (confident exit)",
                action,
                note,
            )
            send_telegram_message(msg)
            return results
        addlog(
            f"[HOLD][{window_name} {window_size}] flat sell condition met but no notes",
            verbose_int=2,
            verbose_state=verbose,
        )
        if verbose >= 3:
            msg = format_window_status(
                symbol,
                window_label,
                trend,
                slope,
                volatility,
                buy_trigger,
                current_pos,
                sell_trigger,
                n_notes,
                "HOLD (confident skip)",
            )
            send_telegram_message(msg)
        return []

    addlog(
        f"[HOLD][{window_name} {window_size}] need={sell_trigger:.2f}, have={sell_p:.2f}, buy_p={buy_p:.2f}",
        verbose_int=2,
        verbose_state=verbose,
    )

    return []

