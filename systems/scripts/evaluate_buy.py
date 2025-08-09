from __future__ import annotations

"""Buy amount evaluator — parity with original SIM engine trigger & sizing."""

def _size_multiplier(score: float) -> float:
    # Same map as sim_engine: 0.5→1x, 1.0→3x  => m = 1 + 4*(score - 0.5)
    # (Scores below 0.5 never trigger.)
    return 1.0 + 4.0 * (score - 0.5)

def evaluate_buy(ctx: dict) -> float:
    """
    Inputs (required):
      should_buy: float in [0..1]  # already computed by engine
      base_unit:  float            # engine-sized base qty

    Returns coin_amount (>= 0.0).
    """
    score = float(ctx["should_buy"])
    if score < 0.5:
        return 0.0
    return float(ctx["base_unit"]) * _size_multiplier(score)
