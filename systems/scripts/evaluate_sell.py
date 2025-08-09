from __future__ import annotations

"""Sell amount evaluator for the simulation engine."""


# API: returns float coin_amount to sell (0 if no sell)
def evaluate_sell(ctx: dict) -> float:
    """
    ctx keys (min):
      topbottom: float
      sell_var: float
      should_sell: float  # 0..1
      base_unit: float
      total_coin: float   # optional guard
    returns: coin_amount (>= 0.0)
    """
    if ctx["should_sell"] < 0.5:
        return 0.0
    m = 1.0 + 4.0 * (ctx["should_sell"] - 0.5)
    # never request more than we own
    want = ctx["base_unit"] * m
    return min(want, max(0.0, ctx.get("total_coin", want)))
