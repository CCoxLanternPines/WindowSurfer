from __future__ import annotations

"""Sell amount evaluator — parity with original SIM engine trigger & sizing."""

def _size_multiplier(score: float) -> float:
    # Same map as sim_engine: 0.5→1x, 1.0→3x
    return 1.0 + 4.0 * (score - 0.5)

def evaluate_sell(ctx: dict) -> float:
    """
    Inputs (required):
      should_sell: float in [0..1]
      base_unit:   float
    Inputs (optional):
      total_coin:  float  # guard so we never request more than we own

    Returns coin_amount (>= 0.0), capped by total_coin if provided.
    """
    score = float(ctx["should_sell"])
    if score < 0.5:
        return 0.0
    want = float(ctx["base_unit"]) * _size_multiplier(score)
    have = float(ctx.get("total_coin", want))
    return max(0.0, min(want, have))
