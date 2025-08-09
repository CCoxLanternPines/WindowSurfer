from __future__ import annotations


def _size_multiplier(score: float) -> float:
    """Legacy ramp: 0.5→1x, 1.0→3x."""
    return 1.0 + 4.0 * (score - 0.5)


def evaluate_buy(ctx: dict) -> float:
    """Return coin amount to buy based on context ``ctx``.

    If ``ctx`` includes ``buy_multiplier`` it is applied directly to the
    ``base_unit``.  Otherwise we fall back to the legacy ramp sizing driven by
    ``should_buy``.
    """
    score = float(ctx["should_buy"])
    if score < 0.5:
        return 0.0

    base = float(ctx["base_unit"])
    mult = ctx.get("buy_multiplier")
    if mult is not None:
        return base * float(mult)
    # Fallback to legacy ramp (keeps old behaviour if knob absent)
    return base * _size_multiplier(score)
