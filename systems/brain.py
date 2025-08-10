from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .paths import (
    BRAINS_DIR,
    TEMP_DIR,
    brain_json,
    ensure_dirs,
    temp_run_dir,
)


def _latest_run_id(tag: str) -> str:
    """Return most recent run id for a given tag."""
    candidates = list(TEMP_DIR.glob(f"*/cluster/centroids_{tag}.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No clustering artifacts found for tag {tag}; run regimes cluster first"
        )
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest.parent.parent.name


def finalize_brain(
    tag: str,
    run_id: str | None,
    labels: Dict[int, str] | None,
    alpha: float = 0.2,
    switch_margin: float = 0.3,
) -> Path:
    """Assemble brain artifact from clustering outputs."""
    ensure_dirs()
    if run_id is None:
        run_id = _latest_run_id(tag)

    run_dir = temp_run_dir(run_id)
    cluster_dir = run_dir / "cluster"
    blocks_dir = run_dir / "blocks"

    cent_path = cluster_dir / f"centroids_{tag}.json"
    assign_path = cluster_dir / f"regime_assignments_{tag}.csv"
    block_plan_path = blocks_dir / f"block_plan_{tag}.json"

    with cent_path.open() as fh:
        centroids = json.load(fh)

    assignments = pd.read_csv(assign_path)
    with block_plan_path.open() as fh:
        block_plan = json.load(fh)
    order_map = {idx + 1: idx for idx, _ in enumerate(block_plan)}
    assignments["_order"] = assignments["block_id"].map(order_map)
    assignments = assignments.sort_values("_order")
    ids = assignments["regime_id"].to_numpy(dtype=int)

    k = int(centroids.get("k", len(centroids["centroids"])))
    counts = np.zeros((k, k), dtype=int)
    for a, b in zip(ids[:-1], ids[1:]):
        counts[a, b] += 1
    transitions = (counts + 1) / (counts.sum(axis=1, keepdims=True) + k)

    brain = {
        "tag": tag,
        "features": centroids["features"],
        "feature_sha": centroids["feature_sha"],
        "scaler": {
            "mean": centroids["mean"],
            "std": centroids["std"],
            "std_floor": centroids.get("std_floor", 1e-6),
        },
        "centroids": centroids["centroids"],
        "k": k,
        "seed": centroids.get("seed", 42),
        "transitions": transitions.tolist(),
        "hysteresis": {"ema_alpha": alpha, "switch_margin": switch_margin},
    }
    if labels:
        brain["labels"] = {str(k): v for k, v in labels.items()}

    path = brain_json(tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(brain, fh, indent=2)
    return path


class RegimeBrain:
    """Lightweight inference helper for regime brains."""

    def __init__(
        self,
        tag: str,
        features: list[str],
        feature_sha: str,
        scaler: Dict[str, list],
        centroids: list[list[float]],
        k: int,
        seed: int,
        transitions: list[list[float]],
        hysteresis: Dict[str, float],
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        self.tag = tag
        self.features = features
        self.feature_sha = feature_sha
        self.scaler = scaler
        self.centroids = np.asarray(centroids, dtype=float)
        self.k = k
        self.seed = seed
        self.transitions = transitions
        self.hysteresis = hysteresis
        self.labels = labels or {}

    @classmethod
    def from_file(cls, path: Path | str) -> "RegimeBrain":
        with open(path) as fh:
            data = json.load(fh)
        return cls(**data)

    def classify(self, features_scaled: np.ndarray) -> int:
        dists = ((self.centroids - features_scaled) ** 2).sum(axis=1)
        return int(dists.argmin())

    def next_probs(self, current_id: int) -> np.ndarray:
        return np.asarray(self.transitions[current_id])

    def blend_knobs(self, p_current: np.ndarray, p_next: np.ndarray):
        pass

