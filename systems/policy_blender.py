from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

# knob ranges from tuner for safety
RANGES: Dict[str, tuple[float, float]] = {
    "position_pct": (0.02, 0.12),
    "max_concurrent": (1, 3),
    "buy_cooldown": (4, 18),
    "volatility_gate": (0.6, 1.2),
    "rsi_buy": (30, 45),
    "take_profit": (0.008, 0.035),
    "trailing_stop": (0.006, 0.03),
    "stop_loss": (0.02, 0.08),
    "sell_cooldown": (3, 16),
}
INT_FIELDS = {"buy_cooldown", "sell_cooldown", "max_concurrent", "rsi_buy"}


def load_seed_knobs() -> Dict[str, Dict[str, Dict[str, float]]]:
    """Load seed knobs from disk."""
    path = Path("regimes/seed_knobs.json")
    with path.open() as fh:
        return json.load(fh)


def classify_current(window_features: np.ndarray, brain) -> np.ndarray:
    """Convert distances to weights based on brain centroids."""
    b = brain._b if hasattr(brain, "_b") else brain
    scaler = b["scaler"]
    mean = np.asarray(scaler["mean"], dtype=float)
    std = np.maximum(np.asarray(scaler["std"], dtype=float), float(scaler.get("std_floor", 1e-6)))
    x = (np.asarray(window_features, dtype=float) - mean) / std
    centroids = np.asarray(b["centroids"], dtype=float)
    dists = np.linalg.norm(centroids - x[None, :], axis=1)
    inv = 1.0 / np.maximum(dists, 1e-9)
    weights = inv / inv.sum() if inv.sum() else np.ones_like(inv) / len(inv)
    return weights


def predict_next(weights: np.ndarray, history: List[int], alpha: float = 0.7) -> np.ndarray:
    k = len(weights)
    trans = np.ones((k, k), float)
    for a, b in zip(history[:-1], history[1:]):
        if 0 <= a < k and 0 <= b < k:
            trans[a, b] += 1.0
    trans = (trans.T / trans.sum(axis=1)).T
    pred = weights @ trans
    blended = alpha * weights + (1 - alpha) * pred
    return blended / blended.sum() if blended.sum() else weights


def apply_hysteresis(weights: np.ndarray, last_top: int | None, boost: float = 0.1) -> np.ndarray:
    w = weights.copy()
    if last_top is not None and int(np.argmax(w)) == last_top:
        w[last_top] += boost
    return w / w.sum() if w.sum() else w


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def blend_knobs(weights: np.ndarray, knobs_by_regime: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    regimes = sorted(knobs_by_regime.keys())
    k = min(len(regimes), len(weights))
    w = np.asarray(weights[:k], dtype=float)
    keys = set().union(*(knobs_by_regime[r].keys() for r in regimes))
    out: Dict[str, float] = {}
    for key in keys:
        vals = [knobs_by_regime[r].get(key, 0.0) for r in regimes[:k]]
        val = float(np.dot(w, vals))
        if key in INT_FIELDS:
            val = int(round(val))
        lo, hi = RANGES.get(key, (None, None))
        if lo is not None and hi is not None:
            val = _clamp(val, lo, hi)
        out[key] = val
    return out


__all__ = [
    "load_seed_knobs",
    "classify_current",
    "predict_next",
    "apply_hysteresis",
    "blend_knobs",
]
