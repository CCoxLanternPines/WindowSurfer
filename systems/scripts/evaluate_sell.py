from __future__ import annotations

"""Sell amount evaluator — parity with original SIM engine trigger & sizing."""

def _size_multiplier(score: float, low_mult: float, high_mult: float) -> float:
    slope = (high_mult - low_mult) / 0.5
    return low_mult + slope * (score - 0.5)

def evaluate_sell(ctx: dict) -> float:
    score = float(ctx["should_sell"])
    if score < 0.5:
        return 0.0
    low_mult = float(ctx.get("sell_mult_low", 1.0))
    high_mult = float(ctx.get("sell_mult_high", 3.0))
    want = float(ctx["base_unit"]) * _size_multiplier(score, low_mult, high_mult)
    have = float(ctx.get("total_coin", want))
    return max(0.0, min(want, have))
