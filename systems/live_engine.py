from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

import numpy as np

from systems.data_loader import load_or_fetch
from systems.brain import RegimeBrain
from systems.features import extract_features, ALL_FEATURES
from systems.simulator import _clean_prices
from systems.policy_blender import (
    load_seed_knobs,
    classify_current,
    predict_next,
    apply_hysteresis,
    blend_knobs,
)

try:  # pragma: no cover - scripts may be absent in tests
    from systems.scripts.evaluate_buy import evaluate_buy  # type: ignore
    from systems.scripts.evaluate_sell import evaluate_sell  # type: ignore
except Exception:  # pragma: no cover
    def evaluate_buy(*args, **kwargs):
        return False

    def evaluate_sell(*args, **kwargs):
        return False


def run_live(
    tag: str,
    *,
    blend_enabled: bool = False,
    alpha: float = 0.7,
    hyst_boost: float = 0.1,
    verbosity: int = 0,
    blend_window: int | None = None,
) -> Dict[str, Any]:
    """Simplified live engine with optional knob blending."""
    candles = _clean_prices(load_or_fetch(tag))
    if blend_enabled and blend_window:
        candles = candles.iloc[-blend_window:].copy()
        print(f"[BLEND] Using last {blend_window} candles for test")

    brain = None
    seed_knobs: Dict[str, Dict[str, Any]] | None = None
    feat_idx: List[int] | None = None
    default_knobs: Dict[str, Any] = {}
    if blend_enabled:
        brain = RegimeBrain.from_file(Path("data/brains") / f"brain_{tag}.json")
        feat_order = brain._b.get("features", [])
        feat_idx = [ALL_FEATURES.index(f) for f in feat_order]
        seed_knobs = load_seed_knobs().get(tag, {})
        if not seed_knobs:
            raise ValueError(f"No seed knobs for tag {tag}")
        default_knobs = next(iter(seed_knobs.values()))
    else:
        seed_all = load_seed_knobs().get(tag, {})
        default_knobs = next(iter(seed_all.values()), {})

    history: List[int] = []
    last_top: int | None = None
    win = 50
    for i in range(len(candles)):
        knobs_now = default_knobs
        if blend_enabled and brain is not None and seed_knobs is not None and i >= win:
            window = candles.iloc[i - win : i]
            feats = extract_features(window)
            window_feats = feats[feat_idx] if feat_idx is not None else feats
            w_now = classify_current(window_feats, brain)
            w_next = predict_next(w_now, history, alpha)
            w_final = apply_hysteresis(w_next, last_top, hyst_boost)
            knobs_now = blend_knobs(w_final, seed_knobs)
            if verbosity >= 3:
                print(
                    f"[BLEND] now={w_now.round(3).tolist()} "
                    f"next={w_next.round(3).tolist()} "
                    f"final={w_final.round(3).tolist()} "
                    f"knobs={knobs_now}"
                )
            top = int(np.argmax(w_final))
            history.append(top)
            if len(history) > 50:
                history.pop(0)
            last_top = top

        price = float(candles.iloc[i]["close"])
        evaluate_buy(price=price, knobs=knobs_now)
        evaluate_sell(price=price, knobs=knobs_now)

    return {"ticks": len(candles)}


__all__ = ["run_live"]
