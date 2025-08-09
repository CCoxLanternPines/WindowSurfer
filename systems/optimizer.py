from __future__ import annotations

from typing import Dict, Any
import numpy as np


def run(
    *,
    trials: int,
    prices: np.ndarray,
    base_policy: Dict[str, Any],
) -> Dict[str, Any]:
    """Placeholder micro-optimizer.

    The function simply returns ``base_policy`` without modification when
    ``trials`` is zero. When ``trials`` is positive a small random nudge is
    applied to numeric fields within ±10% to mimic a tuning step.
    """
    if trials <= 0:
        return base_policy
    rng = np.random.default_rng(0)
    tuned = {}
    for k, v in base_policy.items():
        if isinstance(v, dict):
            tuned[k] = run(trials=trials, prices=prices, base_policy=v)
        elif isinstance(v, (int, float)):
            delta = v * 0.1 * (rng.random() - 0.5)
            tuned[k] = v + delta
        else:
            tuned[k] = v
    return tuned
