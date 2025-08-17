from __future__ import annotations

"""Sell evaluation based on accumulated pressure and slope classification."""

from typing import Any, Dict, List, Tuple

from systems.utils.config import load_settings

CONFIG: Dict[str, Any] = load_settings()

SELL_TRIGGER = 3.0
FLAT_SELL_FRACTION = float(CONFIG.get("flat_sell_fraction", 0.2))
FLAT_SELL_THRESHOLD = float(CONFIG.get("flat_sell_threshold", 0.5))


def evaluate_sell(
    candle: Any,
    slope_cls: int,
    state: Dict[str, Any],
    viz_ax=None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Execute sells once pressure thresholds are crossed.

    Parameters
    ----------
    candle:
        Current market candle.
    slope_cls:
        Classification of the recent slope; typically ``-1``, ``0`` or ``1``.
    state:
        Strategy state shared with the buy evaluator.
    viz_ax:
        Optional Matplotlib axis for plotting sell markers.
    """

    price = float(candle.get("close", 0.0))
    sp = state.get("sell_pressure", 0.0)

    closed: List[Dict[str, Any]] = []
    if state.get("open_notes"):
        if sp >= SELL_TRIGGER:
            # Full liquidation
            closed = state["open_notes"]
            state["open_notes"] = []
            cost = sum(n.get("entry_price", 0.0) for n in closed)
            proceeds = price * len(closed)
            state["realized_pnl"] = state.get("realized_pnl", 0.0) + (proceeds - cost)
            if viz_ax is not None and candle.get("candle_index") is not None:
                viz_ax.scatter(candle["candle_index"], price, color="red", marker="o")
            state["sell_pressure"] = 0.0
        elif slope_cls == 0:
            flat_trigger = SELL_TRIGGER * FLAT_SELL_THRESHOLD
            if sp >= flat_trigger:
                qty = max(1, int(len(state["open_notes"]) * FLAT_SELL_FRACTION))
                closed = state["open_notes"][:qty]
                state["open_notes"] = state["open_notes"][qty:]
                cost = sum(n.get("entry_price", 0.0) for n in closed)
                proceeds = price * qty
                state["realized_pnl"] = state.get("realized_pnl", 0.0) + (proceeds - cost)
                if viz_ax is not None and candle.get("candle_index") is not None:
                    viz_ax.scatter(candle["candle_index"], price, color="orange", marker="o")
                state["sell_pressure"] = 0.0

    return state, closed
