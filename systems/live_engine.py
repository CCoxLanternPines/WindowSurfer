from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

import numpy as np

from systems.data_loader import load_or_fetch
from systems.brain import RegimeBrain
from systems.features import extract_features, ALL_FEATURES
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


def run_blend_live(
    tag: str,
    *,
    alpha: float = 0.7,
    boost: float = 0.1,
    verbosity: int = 0,
) -> Dict[str, Any]:
    """Live-style engine that mirrors simulation blending."""
    candles = load_or_fetch(tag)
    candles = candles.tail(200).reset_index(drop=True)

    brain = RegimeBrain.from_file(Path("data/brains") / f"brain_{tag}.json")
    feat_order = brain._b.get("features", [])
    idx = [ALL_FEATURES.index(f) for f in feat_order]

    seed_knobs = load_seed_knobs().get(tag, {})
    if not seed_knobs:
        raise ValueError(f"No seed knobs for tag {tag}")

    history: List[int] = []
    last_top: int | None = None
    win = 50
    for i in range(win, len(candles)):
        window = candles.iloc[i - win : i]
        feats = extract_features(window)
        window_feats = feats[idx]

        weights_now = classify_current(window_feats, brain)
        weights_next = predict_next(weights_now, history, alpha)
        weights_final = apply_hysteresis(weights_next, last_top, boost)
        blended_knobs = blend_knobs(weights_final, seed_knobs)

        if verbosity >= 3:
            print(
                f"[BLEND] now={weights_now.round(3).tolist()} "
                f"next={weights_next.round(3).tolist()} "
                f"final={weights_final.round(3).tolist()} "
                f"knobs={blended_knobs}"
            )

        price = float(window.iloc[-1]["close"])
        evaluate_buy(price=price, knobs=blended_knobs)
        evaluate_sell(price=price, knobs=blended_knobs)

        top = int(np.argmax(weights_final))
        history.append(top)
        if len(history) > 50:
            history.pop(0)
        last_top = top

    return {"ticks": len(candles) - win}


__all__ = ["run_blend_live"]
