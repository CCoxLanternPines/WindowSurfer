import json
from pathlib import Path
from typing import Dict, Any

import numpy as np

SEED_PATH = Path(__file__).resolve().parent.parent / 'regimes' / 'seed_knobs.json'


def _load_seeds() -> Dict[str, Dict[str, Any]]:
    with SEED_PATH.open() as fh:
        return json.load(fh)


def select_policy(regime_id: str) -> Dict[str, Any]:
    seeds = _load_seeds()
    return seeds.get(regime_id, {})


def _merge(a: Dict[str, Any], b: Dict[str, Any], w: float) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in a:
        if isinstance(a[k], dict):
            out[k] = _merge(a[k], b.get(k, {}), w)
        else:
            av = a[k]
            bv = b.get(k, av)
            out[k] = av * (1 - w) + bv * w
    return out


def blend_policy(distances: np.ndarray, mode: str, blend_alpha: float = 1.0) -> Dict[str, Any]:
    """Blend seed policies based on cluster distances."""
    seeds = _load_seeds()
    ids = list(seeds.keys())
    if mode == 'none' or len(ids) == 1:
        return seeds.get(ids[int(distances.argmin())], {})
    order = distances.argsort()
    if mode == 'top2' and len(order) >= 2:
        i1, i2 = order[:2]
        d1, d2 = distances[i1], distances[i2]
        w2 = 1.0 / (d2 + 1e-9)
        w1 = 1.0 / (d1 + 1e-9)
        w = w2 / (w1 + w2)
        return _merge(seeds[ids[i1]], seeds[ids[i2]], w)
    if mode == 'softmax':
        weights = np.exp(-blend_alpha * distances)
        weights /= weights.sum()
        policy = None
        for idx, w in enumerate(weights):
            r_id = ids[idx]
            if policy is None:
                policy = {k: v for k, v in seeds[r_id].items()}
                continue
            policy = _merge(policy, seeds[r_id], w)
        return policy or {}
    return seeds.get(ids[int(order[0])], {})


class HysteresisRegime:
    """Track regime changes requiring persistence before switching."""

    def __init__(self, hysteresis: int) -> None:
        self.hysteresis = hysteresis
        self.current: str | None = None
        self.pending: str | None = None
        self.count = 0

    def update(self, regime: str) -> str:
        if self.current is None:
            self.current = regime
            return regime
        if regime == self.current:
            self.pending = None
            self.count = 0
            return self.current
        if regime == self.pending:
            self.count += 1
            if self.count >= self.hysteresis:
                self.current = regime
                self.pending = None
                self.count = 0
        else:
            self.pending = regime
            self.count = 1
        return self.current
