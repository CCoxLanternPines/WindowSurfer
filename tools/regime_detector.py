from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

try:  # Optional dependency
    from sklearn.cluster import KMeans  # type: ignore
except Exception:  # pragma: no cover - simple fallback
    KMeans = None


def kmeans_regimes(features: pd.DataFrame, k: int, random_state: int = 0) -> Optional[pd.Series]:
    """Run K-Means on *features* and return regime labels.

    Returns None if k <= 0 or if clustering fails.
    """
    if k <= 0 or features.empty:
        return None
    data = features.dropna()
    if data.empty:
        return None
    # Normalize
    data_norm = (data - data.mean()) / data.std(ddof=0)
    if KMeans is not None:
        model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = model.fit_predict(data_norm.values)
    else:  # pragma: no cover - simple numpy fallback
        from numpy.random import default_rng

        rng = default_rng(random_state)
        centers = data_norm.values[rng.choice(len(data_norm), size=k, replace=False)]
        for _ in range(10):
            dists = np.linalg.norm(data_norm.values[:, None, :] - centers[None, :, :], axis=2)
            labels = dists.argmin(axis=1)
            for j in range(k):
                pts = data_norm.values[labels == j]
                if len(pts) > 0:
                    centers[j] = pts.mean(axis=0)
    return pd.Series(labels, index=data_norm.index, name="regime_label")
