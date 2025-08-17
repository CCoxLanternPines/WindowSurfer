from __future__ import annotations

"""Sell evaluation based on predictive pressures."""

from typing import Any, Dict, List

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
    """Return a list of sell instructions for this candle.

    Each instruction contains:
    ``note``         – the note object to reduce or close
    ``sell_amount``  – coin amount to sell from the note
    ``sell_mode``    – "normal" or "flat"
    """

    runtime_state = runtime_state or {}
    window_name = "strategy"
    strategy = cfg or runtime_state.get("strategy", {})
    window_size = int(strategy.get("window_size", 0))

    verbose = runtime_state.get("verbose", 0)

    pressures = runtime_state.setdefault("pressures", {"buy": {}, "sell": {}})
    sell_p = pressures["sell"].get(window_name, 0.0)
    max_p = strategy.get("max_pressure", 1.0)
    results: List[Dict[str, Any]] = []

    total_size = sum(n.get("entry_amount", 0.0) for n in open_notes)
    if sell_p >= strategy.get("sell_trigger", 0.0) and total_size > 0:
        sell_frac = sell_p / max_p if max_p else 0.0
        trade_size = sell_frac * total_size
        for n in open_notes:
            note_amt = n.get("entry_amount", 0.0)
            amt = sell_frac * note_amt
            if amt > 0:
                results.append({"note": n, "sell_amount": amt, "sell_mode": "normal"})
        pressures["sell"][window_name] = 0.0
        addlog(
            f"[SELL][{window_name} {window_size}] mode=normal size={trade_size:.4f} pressure={sell_p:.1f}/{max_p:.1f}",
            verbose_int=1,
            verbose_state=verbose,
        )
        return results

    slope_cls = runtime_state.get("last_slope_cls", None)
    flat_trigger = strategy.get("sell_trigger", 0.0) * strategy.get("flat_sell_threshold", 1.0)
    if slope_cls == 0 and sell_p >= flat_trigger and total_size > 0:
        frac = strategy.get("flat_sell_fraction", 0.0)
        trade_size = frac * total_size
        for n in open_notes:
            note_amt = n.get("entry_amount", 0.0)
            amt = frac * note_amt
            if amt > 0:
                results.append({"note": n, "sell_amount": amt, "sell_mode": "flat"})
        pressures["sell"][window_name] = 0.0
        addlog(
            f"[SELL][{window_name} {window_size}] mode=flat size={trade_size:.4f} pressure={sell_p:.1f}/{max_p:.1f}",
            verbose_int=1,
            verbose_state=verbose,
        )
        return results

    return []
