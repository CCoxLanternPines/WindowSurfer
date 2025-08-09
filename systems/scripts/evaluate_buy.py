from __future__ import annotations

"""Buy amount evaluator — parity with original SIM engine trigger & sizing."""

def _size_multiplier(score: float, low_mult: float, high_mult: float) -> float:
    # Linear map: score=0.5 → low_mult, score=1.0 → high_mult
    slope = (high_mult - low_mult) / 0.5
    return low_mult + slope * (score - 0.5)

def evaluate_buy(ctx: dict) -> float:
    score = float(ctx["should_buy"])
    if score < 0.5:
        return 0.0
    low_mult = float(ctx.get("buy_mult_low", 1.0))
    high_mult = float(ctx.get("buy_mult_high", 3.0))
    return float(ctx["base_unit"]) * _size_multiplier(score, low_mult, high_mult)
