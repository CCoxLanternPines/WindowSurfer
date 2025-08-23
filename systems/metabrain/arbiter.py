from __future__ import annotations

from .rules import buy_decision, sell_decision


def run_arbiter(features: dict, position_state: str, debug: bool = False):
    reasons = []
    if position_state == "flat":
        buy, why = buy_decision(features, debug=debug)
        if buy:
            return "BUY", why
        reasons.extend(why)

    if position_state == "long":
        sell, why = sell_decision(features, debug=debug)
        if sell:
            return "SELL", why
        reasons.extend(why)

    return "HOLD", reasons

