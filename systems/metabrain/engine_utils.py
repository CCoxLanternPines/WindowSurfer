from __future__ import annotations

import importlib

from .registry import QUESTION_REGISTRY

BRAIN_MODULES = [
    "exhaustion",
    "reversal",
    "momentum_inflection",
    "bottom_catcher",
    "divergence",
    "rolling_peak",
]

def cache_all_brains(df):
    """Run all brains once across full df, store per-candle signals + summaries."""
    brain_cache: dict[str, dict[str, object]] = {}
    for mod_name in BRAIN_MODULES:
        mod = importlib.import_module(f"systems.brains.{mod_name}")
        signals = mod.run(df, viz=False)
        summary = mod.summarize(signals, df)
        brain_cache[mod_name] = {"signals": signals, "summary": summary}
    return brain_cache

def extract_features_at_t(brain_cache, t):
    """Extract feature dict at candle t from cached brain signals."""
    features: dict[str, object] = {}

    for qid, (brain, key) in QUESTION_REGISTRY.items():
        val = brain_cache.get(brain, {}).get("summary", {}).get("stats", {}).get(key)
        features[qid] = val

    for brain, data in brain_cache.items():
        valid = []
        for s in data["signals"]:
            idx = s.get("index", s.get("candle_index"))
            if idx is None:
                continue
            if idx <= t:
                valid.append((idx, s))
        if not valid:
            continue
        _, last_signal = max(valid, key=lambda x: x[0])
        stats = data["summary"]
        brain_name = stats.get("brain", brain)
        for k, v in last_signal.items():
            if k not in ("index", "candle_index"):
                features[f"{brain_name}_{k}"] = v
    return features
