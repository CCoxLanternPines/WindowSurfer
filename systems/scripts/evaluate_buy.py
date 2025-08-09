from __future__ import annotations

def evaluate_buy(ctx: dict) -> float:
    score = float(ctx["should_buy"])
    if score < 0.5:
        return 0.0

    base = float(ctx["base_unit"])
    mult = float(ctx.get("buy_multiplier", 1.0))

    # Optional ladder sizing (clean + defaults)
    rules = ctx.get("buy_rules", {}) or {}
    ladder_tb   = list((rules.get("ladder_tb") or []))
    ladder_mult = list((rules.get("ladder_mult") or []))

    # Gates (optional)
    gate_mom  = bool(rules.get("gate_momentum", False))
    gate_wick = bool(rules.get("gate_wick_bias", False))
    if gate_mom and float(ctx.get("slope_micro", 0.0)) >= 0.0:
        return 0.0  # momentum not down → skip buy
    if gate_wick and float(ctx.get("lw_ratio", 0.0)) <= float(ctx.get("uw_ratio", 0.0)):
        return 0.0  # no lower-wick dominance → skip buy

    # Pick deepest matching ladder multiplier
    tb = float(ctx.get("topbottom", 0.5))
    ladder = 1.0
    if ladder_tb and ladder_mult and len(ladder_tb) == len(ladder_mult):
        for th, m in zip(ladder_tb, ladder_mult):
            if tb <= float(th):
                ladder = float(m)

    return base * mult * ladder
