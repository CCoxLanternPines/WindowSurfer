from __future__ import annotations

"""Buy amount evaluator — parity with original SIM engine trigger & sizing."""

def evaluate_buy(ctx: dict) -> float:
    score = float(ctx["should_buy"])
    if score < 0.5:
        return 0.0
    base = float(ctx["base_unit"])
    mult = float(ctx.get("buy_multiplier", 1.0))
    return base * mult
