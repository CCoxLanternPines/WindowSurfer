from __future__ import annotations

from .registry import QUESTION_REGISTRY


def extract_features(all_brains: dict) -> dict:
    features = {}
    for qid, (brain, key) in QUESTION_REGISTRY.items():
        val = all_brains.get(brain, {}).get("stats", {}).get(key, None)
        features[qid] = val

    regime_stats = all_brains.get("regime", {}).get("stats", {})
    if regime_stats:
        features["regime"] = regime_stats
        trend = regime_stats.get("trend", "")
        vol = regime_stats.get("vol", "")
        features["regime_key"] = f"{trend}.{vol}"

    return features
