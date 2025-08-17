import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from systems.scripts.strategy_pressure import (
    pressure_buy_signal,
    pressure_sell_signal,
    pressure_flat_sell_signal,
)


def _ref_pressure_buy_signal(candle, state):
    price = float(candle.get("close", 0.0))
    anchor = float(state.get("anchor_price", price))
    pressure = float(state.get("pressure", 0.0))
    drop_scale = float(state.get("drop_scale", 0.005))
    trigger = anchor * (1.0 - pressure * drop_scale)
    return price <= trigger


def _ref_pressure_sell_signal(candle, note, state):
    price = float(candle.get("close", 0.0))
    buy_price = float(note.get("price", 0.0))
    pressure = float(state.get("pressure", 0.0))
    base_profit = float(state.get("base_profit", 0.01))
    pressure_scale = float(state.get("pressure_scale", 0.01))
    target_gain = base_profit + pressure * pressure_scale
    gain = (price - buy_price) / buy_price if buy_price else 0.0
    return gain >= target_gain


def _ref_pressure_flat_sell_signal(candle, state):
    price = float(candle.get("close", 0.0))
    anchor = float(state.get("anchor_price", price))
    drawdown = float(state.get("flat_sell_drawdown", 0.03))
    trigger = anchor * (1.0 - drawdown)
    return price <= trigger


def test_pressure_buy_parity():
    state = {"anchor_price": 100.0, "pressure": 1.0, "drop_scale": 0.01}
    candle_hit = {"close": 98.0}
    candle_miss = {"close": 99.5}
    assert pressure_buy_signal(candle_hit, state) == _ref_pressure_buy_signal(candle_hit, state)
    assert pressure_buy_signal(candle_miss, state) == _ref_pressure_buy_signal(candle_miss, state)


def test_pressure_sell_parity():
    note = {"price": 100.0}
    state = {"pressure": 2.0, "base_profit": 0.02, "pressure_scale": 0.01}
    candle_hit = {"close": 104.0}
    candle_miss = {"close": 103.0}
    assert pressure_sell_signal(candle_hit, note, state) == _ref_pressure_sell_signal(candle_hit, note, state)
    assert pressure_sell_signal(candle_miss, note, state) == _ref_pressure_sell_signal(candle_miss, note, state)


def test_pressure_flat_sell_parity():
    state = {"anchor_price": 100.0, "flat_sell_drawdown": 0.05}
    candle_hit = {"close": 94.0}
    candle_miss = {"close": 96.0}
    assert pressure_flat_sell_signal(candle_hit, state) == _ref_pressure_flat_sell_signal(candle_hit, state)
    assert pressure_flat_sell_signal(candle_miss, state) == _ref_pressure_flat_sell_signal(candle_miss, state)
