from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .data_loader import _load_settings
from .features import _feature_sha


def cluster_features(
    features_df: pd.DataFrame, meta: Dict[str, list], k: int, seed: int = 0, max_iter: int = 100
) -> Tuple[pd.DataFrame, Dict[str, list], float]:
    """Cluster standardized features using K-Means.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame with scaled feature columns and ``block_id``.
    meta : dict
        Metadata containing ``mean``, ``std`` and ``features``.
    k : int
        Number of clusters.
    seed : int, optional
        RNG seed for deterministic initialization.
    max_iter : int, optional
        Maximum K-Means iterations.
    """
    feature_names = meta["features"]
    sha_meta = meta.get("feature_sha")
    if sha_meta and sha_meta != _feature_sha(feature_names):
        raise ValueError("[FEATURES][FATAL] feature_sha mismatch; regenerate features")

    settings = _load_settings()
    cfg = settings.get("cluster_settings", {})
    drops = set(cfg.get("drop_features", []))
    replacements = cfg.get("replace_features", {})
    feat_names_effective = [replacements.get(f, f) for f in feature_names if f not in drops]

    X = features_df[feat_names_effective].to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(X), size=k, replace=False)
    centroids = X[indices]

    for _ in range(max_iter):
        distances = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)
        new_centroids = np.array(
            [X[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j] for j in range(k)]
        )
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids

    distances = ((X - centroids[labels]) ** 2).sum(axis=1)
    inertia = float(distances.sum())

    assignments_df = pd.DataFrame({"block_id": features_df["block_id"], "regime_id": labels})

    mean = [meta["mean"][meta["features"].index(f)] for f in feat_names_effective]
    std = [meta["std"][meta["features"].index(f)] for f in feat_names_effective]
    centroids_scaled = centroids.tolist()
    centroids_payload = {
        "features": feat_names_effective,
        "feature_sha": _feature_sha(feat_names_effective),
        "mean": mean,
        "std": std,
        "std_floor": meta.get("std_floor", 1e-6),
        "centroids": centroids_scaled,
        "k": k,
        "inertia": float(inertia),
    }
    return assignments_df, centroids_payload, inertia


def align_centroids(meta: Dict[str, list], centroids: Dict[str, list]) -> Dict[str, list]:
    """Verify and realign centroids against feature meta."""
    if centroids.get("feature_sha") != _feature_sha(centroids.get("features", [])):
        raise ValueError("[CLUSTER] Corrupt centroids: internal feature_sha mismatch.")

    sha_meta = meta.get("feature_sha")
    sha_cent = centroids.get("feature_sha")
    if sha_meta == sha_cent and meta["features"] == centroids["features"]:
        aligned_features = meta["features"]
    else:
        meta_set = set(meta["features"])
        cent_set = set(centroids["features"])
        common = [f for f in meta["features"] if f in cent_set]
        dropped_meta = [f for f in meta["features"] if f not in cent_set]
        dropped_cent = [f for f in centroids["features"] if f not in meta_set]
        print(
            f"[ALIGN] Realigning features; dropping only-in-meta: {dropped_meta}, "
            f"only-in-centroids: {dropped_cent}"
        )
        mean = [meta["mean"][meta["features"].index(f)] for f in common]
        std = [meta["std"][meta["features"].index(f)] for f in common]
        cent_order = [centroids["features"].index(f) for f in common]
        C = np.asarray(centroids["centroids"] )[:, cent_order]
        centroids = {
            **centroids,
            "features": common,
            "centroids": C.tolist(),
            "mean": mean,
            "std": std,
            "feature_sha": _feature_sha(common),
        }
        aligned_features = common

    if len(aligned_features) < 6:
        raise ValueError(
            f"[ALIGN][FATAL] Too few common features after realign: {len(aligned_features)}"
        )
    return centroids
