from __future__ import annotations

from .rules import buy_decision, sell_decision


def run_arbiter(features: dict, position_state: str) -> str:
    if position_state == "flat" and buy_decision(features):
        return "BUY"
    if position_state == "long" and sell_decision(features):
        return "SELL"
    return "HOLD"

