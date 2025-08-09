from __future__ import annotations

"""Sell amount evaluator — parity with original SIM engine trigger & sizing."""

def evaluate_sell(ctx: dict) -> float:
    score = float(ctx["should_sell"])
    if score < 0.5:
        return 0.0
    base = float(ctx["base_unit"])
    mult = float(ctx.get("sell_multiplier", 1.0))
    want = base * mult
    have = float(ctx.get("total_coin", want))
    return max(0.0, min(want, have))
