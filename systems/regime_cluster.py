from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Tuple


def cluster_features(
    features_df: pd.DataFrame, meta: Dict[str, list], k: int, seed: int = 0, max_iter: int = 100
) -> Tuple[pd.DataFrame, Dict[str, list], float]:
    """Cluster standardized features using K-Means.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame with unscaled feature columns and ``block_id``.
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
    mean = np.asarray(meta["mean"], dtype=float)
    std = np.asarray(meta["std"], dtype=float)

    X_raw = features_df[feature_names].to_numpy(dtype=float)
    X = (X_raw - mean) / std

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
    centroids_dict = {
        "features": feature_names,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "centroids": centroids.tolist(),
    }
    return assignments_df, centroids_dict, inertia
