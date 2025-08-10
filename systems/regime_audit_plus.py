import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .regime_cluster import align_centroids
from .paths import temp_audit_dir


def run(tag: str, paths: Dict[str, Path], run_id: str, verbose: int = 0) -> Dict[str, Path | float | dict]:
    """Perform extended audit on regime artifacts."""
    features_df = pd.read_parquet(paths["features"])
    with Path(paths["meta"]).open() as fh:
        meta = json.load(fh)
    assignments = pd.read_csv(paths["assignments"])
    with Path(paths["centroids"]).open() as fh:
        centroids = json.load(fh)
    with Path(paths["block_plan"]).open() as fh:
        block_plan = json.load(fh)

    centroids = align_centroids(meta, centroids)
    feat = centroids["features"]
    mean = np.asarray(centroids["mean"], dtype=float)
    std = np.asarray(centroids["std"], dtype=float)

    C_scaled = np.asarray(centroids["centroids"], dtype=float)
    C_unscaled = C_scaled * std + mean

    Xs = features_df[feat].to_numpy(dtype=float)
    Xu = np.nan_to_num(Xs * std + mean)
    df_unscaled = pd.DataFrame(Xu, columns=feat)
    df_unscaled.insert(0, "block_id", features_df["block_id"].to_numpy())
    merged = assignments.merge(df_unscaled, on="block_id")

    reg_means_df = merged.groupby("regime_id")[feat].mean().reindex(range(C_unscaled.shape[0]))
    reg_means = np.nan_to_num(reg_means_df.to_numpy())
    max_diff = float(np.max(np.abs(reg_means - C_unscaled)))

    audit_dir = temp_audit_dir(run_id)
    audit_dir.mkdir(parents=True, exist_ok=True)

    cent_df = pd.DataFrame({"regime_id": range(C_unscaled.shape[0])})
    for i, f in enumerate(feat):
        cent_df[f"centroid_{f}"] = C_unscaled[:, i]
        cent_df[f"mean_{f}"] = reg_means[:, i]
    cent_path = audit_dir / f"centroids_unscaled_{tag}.csv"
    cent_df.to_csv(cent_path, index=False)

    global_mean = Xu.mean(axis=0)
    cluster_ids = assignments["regime_id"].to_numpy()
    K = C_unscaled.shape[0]
    n_k = np.array([(cluster_ids == k).sum() for k in range(K)], dtype=float)

    SSB = (n_k[:, None] * (reg_means - global_mean) ** 2).sum(axis=0)
    SSW = np.zeros_like(global_mean)
    for k in range(K):
        Xk = Xu[cluster_ids == k]
        if Xk.size:
            diff = Xk - reg_means[k]
            SSW += (diff**2).sum(axis=0)
    F_scores = SSB / np.maximum(SSW, 1e-12)

    fi_df = pd.DataFrame(
        {
            "feature": feat,
            "F_score": F_scores,
            "global_mean": global_mean,
        }
    )
    for k in range(K):
        fi_df[f"cluster_{k}_mean"] = reg_means[k]
    fi_path = audit_dir / f"feature_influence_{tag}.csv"
    fi_df.to_csv(fi_path, index=False)

    top_records = []
    for k in range(K):
        deltas = reg_means[k] - global_mean
        order = np.argsort(np.abs(deltas))[::-1]
        for rank, idx in enumerate(order[:3], start=1):
            top_records.append(
                {
                    "regime_id": k,
                    "rank": rank,
                    "feature": feat[idx],
                    "delta_unscaled": deltas[idx],
                    "cluster_mean": reg_means[k, idx],
                    "global_mean": global_mean[idx],
                    "F_score": F_scores[idx],
                }
            )
    top_df = pd.DataFrame(top_records)
    top_path = audit_dir / f"top_drivers_{tag}.csv"
    top_df.to_csv(top_path, index=False)

    bp_df = pd.DataFrame(block_plan)
    bp_df.insert(0, "block_id", np.arange(1, len(bp_df) + 1))
    assign_dates = assignments.merge(
        bp_df[["block_id", "train_start", "train_end"]], on="block_id", how="left"
    )
    assign_path = audit_dir / f"assignments_with_dates_{tag}.csv"
    assign_dates.to_csv(assign_path, index=False)

    spans = []
    for rid, grp in assign_dates.sort_values("block_id").groupby("regime_id"):
        blk_ids = grp["block_id"].to_numpy()
        starts = grp["train_start"].to_numpy()
        ends = grp["train_end"].to_numpy()
        s = starts[0]
        e = ends[0]
        count = 1
        for i in range(1, len(blk_ids)):
            if blk_ids[i] == blk_ids[i - 1] + 1:
                e = ends[i]
                count += 1
            else:
                spans.append({"regime_id": rid, "start_date": s, "end_date": e, "num_blocks": count})
                s = starts[i]
                e = ends[i]
                count = 1
        spans.append({"regime_id": rid, "start_date": s, "end_date": e, "num_blocks": count})
    span_df = pd.DataFrame(spans)
    span_path = audit_dir / f"regime_spans_{tag}.csv"
    span_df.to_csv(span_path, index=False)

    counts = assignments["regime_id"].value_counts().sort_index().to_dict()
    top_global = fi_df.sort_values("F_score", ascending=False).head(3)["feature"].tolist()
    print(f"[AUDIT++] K={K} | sizes: {counts}")
    print(f"[AUDIT++] Max centroid diff (empirical vs model, unscaled): {max_diff:.6f}")
    print(f"[AUDIT++] Top global drivers (F-score): {', '.join(top_global)}")
    for k in range(K):
        deltas = reg_means[k] - global_mean
        order = np.argsort(np.abs(deltas))[::-1][:3]
        parts = []
        for idx in order:
            sign = '+' if deltas[idx] >= 0 else '-'
            parts.append(f"{sign}{feat[idx]}")
        print(f"[AUDIT++] R{k} top deltas: {', '.join(parts)}")
    print(
        "[AUDIT++] Files: centroids_unscaled.csv, feature_influence.csv, top_drivers.csv, "
        "assignments_with_dates.csv, regime_spans.csv"
    )

    return {
        "centroids_unscaled": cent_path,
        "feature_influence": fi_path,
        "top_drivers": top_path,
        "assignments_with_dates": assign_path,
        "regime_spans": span_path,
        "max_centroid_diff": max_diff,
        "sizes": counts,
        "top_global_drivers": top_global,
    }

