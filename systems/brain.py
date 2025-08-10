from __future__ import annotations

from pathlib import Path
import json, numpy as np, pandas as pd
from datetime import datetime
from systems.paths import BRAINS_DIR, temp_cluster_dir, temp_audit_dir
from systems.paths import load_settings


def _nowstamp():
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def finalize_brain(tag: str, run_id: str, labels: dict[int, str] | None = None,
                   alpha: float = 0.2, switch_margin: float = 0.3) -> Path:
    # 1) Load aligned centroids payload (already includes features, mean, std, std_floor)
    c_dir = temp_cluster_dir(run_id)
    centroids_path = max(c_dir.glob("centroids_*.json"))  # newest
    C = json.loads(Path(centroids_path).read_text())
    feat = C["features"]; sha = C["feature_sha"]
    mean = np.asarray(C["mean"], float).tolist()
    std = np.asarray(C["std"], float).tolist()
    std_floor = float(C.get("std_floor", 1e-6))
    centroids = C["centroids"]; k = int(C["k"]); seed = int(C.get("seed", 42))

    # 2) Build Markov P(next|current) from assignments_with_dates
    a_dir = temp_audit_dir(run_id)
    assign_path = max(a_dir.glob("assignments_with_dates_*.csv"))
    df = pd.read_csv(assign_path).sort_values("block_id")
    trans = np.ones((k, k), float)  # Laplace smoothing init
    for (cur, nxt) in zip(df.regime_id.values[:-1], df.regime_id.values[1:]):
        if 0 <= cur < k and 0 <= nxt < k:
            trans[cur, nxt] += 1.0
    trans = (trans.T / trans.sum(axis=1)).T  # row-stochastic

    # 3) Labels (optional)
    label_map = {str(i): labels.get(i, f"Regime {i}") for i in range(k)} if labels else {}

    # 4) Brain payload
    brain = {
        "tag": tag,
        "k": k,
        "seed": seed,
        "features": feat,
        "feature_sha": sha,
        "scaler": {"mean": mean, "std": std, "std_floor": std_floor},
        "centroids": centroids,       # scaled space
        "labels": label_map,
        "transitions": trans.tolist(),
        "hysteresis": {"ema_alpha": alpha, "switch_margin": switch_margin},
        "meta": {"run_id": run_id, "created_utc": _nowstamp()}
    }

    # 5) Save versioned brain file
    out = BRAINS_DIR / f"brain_{tag}__k{k}_seed{seed}_{_nowstamp()}.json"
    BRAINS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(brain, indent=2))

    return out


def write_latest_copy(path: Path, tag: str) -> Path:
    latest = BRAINS_DIR / f"brain_{tag}.json"
    latest.write_text(Path(path).read_text())
    return latest


class RegimeBrain:
    @staticmethod
    def from_file(path: str | Path) -> "RegimeBrain":
        obj = json.loads(Path(path).read_text())
        rb = RegimeBrain(); rb._b = obj; return rb

    def classify_now(self, x: np.ndarray) -> int:
        s = self._b["scaler"]
        mean = np.asarray(s["mean"])
        std = np.asarray(s["std"])
        std_floor = float(s.get("std_floor", 1e-6))
        x_scaled = (np.asarray(x) - mean) / np.maximum(std, std_floor)
        return self.classify_scaled(x_scaled)

    def classify_scaled(self, x_scaled: np.ndarray) -> int:
        # x_scaled: shape (F,)
        import numpy as np
        C = np.asarray(self._b["centroids"])
        d = ((C - x_scaled[None, :])**2).sum(axis=1)
        return int(np.argmin(d))

    def next_probs(self, current_id: int) -> list[float]:
        P = np.asarray(self._b["transitions"])
        return P[current_id].tolist()

