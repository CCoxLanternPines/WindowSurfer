from __future__ import annotations

from .registry import QUESTION_REGISTRY


def extract_features(all_brains: dict) -> dict:
    features = {}
    for qid, (brain, key) in QUESTION_REGISTRY.items():
        val = all_brains.get(brain, {}).get("stats", {}).get(key, None)
        features[qid] = val
    return features

