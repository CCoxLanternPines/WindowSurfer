from __future__ import annotations

"""Pressure-based trading signal helpers."""

from typing import Any, Dict

from systems.utils.addlog import addlog


def pressure_buy_signal(candle: Dict[str, float], state: Dict[str, Any]) -> bool:
    """Return True if conditions trigger a pressure buy.

    Parameters
    ----------
    candle:
        Dictionary representing current candle with at least ``close``.
    state:
        Mutable state providing ``anchor_price``, ``pressure`` and config knobs.
    """
    price = float(candle.get("close", 0.0))
    anchor = float(state.get("anchor_price", price))
    pressure = float(state.get("pressure", 0.0))
    drop_scale = float(state.get("drop_scale", 0.005))
    trigger = anchor * (1.0 - pressure * drop_scale)
    decision = price <= trigger
    if decision and state.get("debug"):
        addlog(
            f"[PRESSURE_BUY] price={price:.2f} trigger={trigger:.2f} "
            f"anchor={anchor:.2f} pressure={pressure:.3f}",
            verbose_state=state.get("verbose", 0),
        )
    return decision


def pressure_sell_signal(
    candle: Dict[str, float], note: Dict[str, Any], state: Dict[str, Any]
) -> bool:
    """Return True if conditions trigger a pressure-scaled sell."""
    price = float(candle.get("close", 0.0))
    buy_price = float(note.get("price", 0.0))
    pressure = float(state.get("pressure", 0.0))
    base_profit = float(state.get("base_profit", 0.01))
    pressure_scale = float(state.get("pressure_scale", 0.01))
    target_gain = base_profit + pressure * pressure_scale
    gain = (price - buy_price) / buy_price if buy_price else 0.0
    decision = gain >= target_gain
    if decision and state.get("debug"):
        addlog(
            f"[PRESSURE_SELL] price={price:.2f} gain={gain:.4f} "
            f"target={target_gain:.4f}",
            verbose_state=state.get("verbose", 0),
        )
    return decision


def pressure_flat_sell_signal(candle: Dict[str, float], state: Dict[str, Any]) -> bool:
    """Return True if conditions trigger a flat sell (stop-loss)."""
    price = float(candle.get("close", 0.0))
    anchor = float(state.get("anchor_price", price))
    drawdown = float(state.get("flat_sell_drawdown", 0.03))
    trigger = anchor * (1.0 - drawdown)
    decision = price <= trigger
    if decision and state.get("debug"):
        addlog(
            f"[FLAT_SELL] price={price:.2f} trigger={trigger:.2f} "
            f"anchor={anchor:.2f} drawdown={drawdown:.3f}",
            verbose_state=state.get("verbose", 0),
        )
    return decision
