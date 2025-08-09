from __future__ import annotations


def _size_multiplier(score: float) -> float:
    """Legacy ramp: 0.5→1x, 1.0→3x."""
    return 1.0 + 4.0 * (score - 0.5)


def evaluate_sell(ctx: dict) -> float:
    """Return coin amount to sell based on context ``ctx``.

    ``sell_multiplier`` directly controls the size when present.  Otherwise the
    legacy ramp is used as a fallback.  Never returns more coin than available
    in ``total_coin``.
    """
    score = float(ctx["should_sell"])
    if score < 0.5:
        return 0.0

    base = float(ctx["base_unit"])
    mult = ctx.get("sell_multiplier")
    if mult is not None:
        want = base * float(mult)
    else:
        want = base * _size_multiplier(score)

    have = float(ctx.get("total_coin", want))
    return max(0.0, min(want, have))
