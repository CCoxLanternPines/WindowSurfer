import json
from pathlib import Path
import numpy as np
from .features import zscore_features

REGIME_DIR = Path(__file__).resolve().parent.parent / 'data' / 'tmp' / 'regimes'
REGIME_DIR.mkdir(parents=True, exist_ok=True)


def fit_kmeans(X_train: np.ndarray, k: int) -> dict:
    """Run a simple k-means clustering on ``X_train``."""
    Xn, stats = zscore_features(X_train)
    rng = np.random.default_rng(0)
    centroids = Xn[rng.choice(len(Xn), size=k, replace=False)]
    for _ in range(10):
        dists = np.linalg.norm(Xn[:, None, :] - centroids[None, :, :], axis=2)
        labels = dists.argmin(axis=1)
        for i in range(k):
            members = Xn[labels == i]
            if len(members):
                centroids[i] = members.mean(axis=0)
    return {"centroids": centroids.tolist(), "scale": stats}


def save_centroids(tag: str, k: int, model: dict) -> None:
    path = REGIME_DIR / f"{tag}_k{k}.json"
    with path.open('w') as fh:
        json.dump(model, fh)


def assign_regime(x: np.ndarray, model: dict) -> tuple[int, np.ndarray]:
    centroids = np.array(model['centroids'])
    x_scaled, _ = zscore_features(x[None, :], model['scale'])
    dists = np.linalg.norm(centroids - x_scaled, axis=1)
    idx = int(dists.argmin())
    return idx, dists
