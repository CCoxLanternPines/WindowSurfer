from __future__ import annotations

"""Buy amount evaluator for the simulation engine."""


# API: returns float coin_amount to buy (0 if no buy)
def evaluate_buy(ctx: dict) -> float:
    """
    ctx keys (min):
      topbottom: float
      buy_var: float
      should_buy: float  # 0..1
      base_unit: float
    returns: coin_amount (>= 0.0)
    """
    if ctx["should_buy"] < 0.5:
        return 0.0
    # multiplier 0.5→1x, 1.0→3x (match current sim_engine logic)
    m = 1.0 + 4.0 * (ctx["should_buy"] - 0.5)
    return ctx["base_unit"] * m
