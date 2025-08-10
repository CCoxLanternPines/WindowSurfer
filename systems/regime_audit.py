from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .regime_cluster import align_centroids
from .paths import temp_audit_dir


def run_audit(tag: str, paths: Dict[str, Path], run_id: str, verbose: int = 0) -> None:
    """Run audit on existing regime clustering artifacts."""
    features_df = pd.read_parquet(paths["features"])
    with Path(paths["meta"]).open() as fh:
        meta = json.load(fh)
    assignments = pd.read_csv(paths["assignments"])
    with Path(paths["centroids"]).open() as fh:
        centroids_info = json.load(fh)
    with Path(paths["block_plan"]).open() as fh:
        block_plan = json.load(fh)

    centroids_info = align_centroids(meta, centroids_info)
    feat = centroids_info["features"]
    mean = np.asarray(centroids_info["mean"], dtype=float)
    std = np.asarray(centroids_info["std"], dtype=float)

    centroids_scaled = np.asarray(centroids_info["centroids"], dtype=float)
    centroids_unscaled = centroids_scaled * std + mean

    Xs = features_df[feat].to_numpy(dtype=float)
    Xu = Xs * std + mean
    df_unscaled = pd.DataFrame(Xu, columns=feat)
    df_unscaled.insert(0, "block_id", features_df["block_id"].to_numpy())
    merged_unscaled = assignments.merge(df_unscaled, on="block_id")
    reg_means = merged_unscaled.groupby("regime_id")[feat].mean().to_numpy()
    max_diff = float(np.max(np.abs(reg_means - centroids_unscaled)))

    global_mean = Xu.mean(axis=0)

    counts = assignments["regime_id"].value_counts().sort_index()
    K = centroids_scaled.shape[0]
    print(f"[AUDIT] K={K} | sizes: {counts.to_dict()}")
    print(f"[AUDIT] Centroid sanity max abs diff: {max_diff:.6f}")

    audit_dir = temp_audit_dir(run_id)
    audit_dir.mkdir(parents=True, exist_ok=True)

    cent_df = pd.DataFrame(centroids_unscaled, columns=feat)
    cent_df.insert(0, "regime_id", range(K))
    cent_path = audit_dir / f"centroids_unscaled_{tag}.csv"
    cent_df.to_csv(cent_path, index=False)
    if verbose >= 3:
        print(cent_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    deltas_records = []
    feature_idx = {f: i for i, f in enumerate(feat)}
    for r in range(K):
        delta = centroids_unscaled[r] - global_mean
        top_idx = np.argsort(np.abs(delta))[::-1][:3]
        for idx in top_idx:
            deltas_records.append(
                {
                    "regime_id": r,
                    "feature": feat[idx],
                    "delta_unscaled": delta[idx],
                    "centroid_value": centroids_unscaled[r, idx],
                    "global_mean": global_mean[idx],
                }
            )
    top_df = pd.DataFrame(deltas_records)
    top_path = audit_dir / f"top_features_{tag}.csv"
    top_df.to_csv(top_path, index=False)

    dist_matrix = np.sqrt(
        ((centroids_scaled[:, None, :] - centroids_scaled[None, :, :]) ** 2).sum(axis=2)
    )
    print(f"[AUDIT] Inter-centroid distances (scaled): {np.round(dist_matrix, 2).tolist()}")

    merged_scaled = assignments.merge(features_df, on="block_id")
    intra_variance = {}
    for r in range(K):
        Xr = merged_scaled[merged_scaled["regime_id"] == r][feat].to_numpy(dtype=float)
        if len(Xr):
            var = float(((Xr - centroids_scaled[r]) ** 2).sum(axis=1).mean())
        else:
            var = float("nan")
        intra_variance[r] = var
    var_str = ", ".join(f"R{r}={v:.3f}" for r, v in intra_variance.items())
    print(f"[AUDIT] Intra variance (per regime): {var_str}")

    quality_df = pd.DataFrame(
        {
            "regime_id": range(K),
            "size": [counts.get(i, 0) for i in range(K)],
            "intra_variance": [intra_variance.get(i, np.nan) for i in range(K)],
        }
    )
    for j in range(K):
        quality_df[f"dist_to_{j}"] = dist_matrix[:, j]
    qual_path = audit_dir / f"regime_quality_{tag}.csv"
    quality_df.to_csv(qual_path, index=False)

    vol_idx = feature_idx.get("volatility")
    volvol_idx = feature_idx.get("vol_of_vol")
    bb_idx = feature_idx.get("bb_width_avg")

    vol_med = np.median(Xu[:, vol_idx]) if vol_idx is not None else np.nan
    volvol_med = np.median(Xu[:, volvol_idx]) if volvol_idx is not None else np.nan
    bb_third = np.quantile(Xu[:, bb_idx], 1 / 3) if bb_idx is not None else np.nan

    labels = []
    for r in range(K):
        ps = (
            centroids_unscaled[r, feature_idx["price_slope"]]
            if "price_slope" in feature_idx
            else np.nan
        )
        mb = (
            centroids_unscaled[r, feature_idx["ma_bias"]]
            if "ma_bias" in feature_idx
            else np.nan
        )
        dd = (
            centroids_unscaled[r, feature_idx["avg_dd_depth"]]
            if "avg_dd_depth" in feature_idx
            else np.nan
        )
        vol = centroids_unscaled[r, vol_idx] if vol_idx is not None else np.nan
        volvol = centroids_unscaled[r, volvol_idx] if volvol_idx is not None else np.nan
        bb = centroids_unscaled[r, bb_idx] if bb_idx is not None else np.nan
        pct = (
            centroids_unscaled[r, feature_idx["pct_inside_1std"]]
            if "pct_inside_1std" in feature_idx
            else np.nan
        )

        label = "Transition"
        if np.all([ps > 0, mb > 0.55, dd < 0.08]):
            label = "Bull"
        else:
            vol_high = False
            if not np.isnan(vol) and not np.isnan(vol_med):
                vol_high = vol >= vol_med
            if not np.isnan(volvol) and not np.isnan(volvol_med):
                vol_high = vol_high or (volvol >= volvol_med)
            if (ps < 0) and (dd >= 0.10) and vol_high:
                label = "Bear"
            elif (not np.isnan(bb) and not np.isnan(bb_third) and bb < bb_third) and (
                not np.isnan(pct) and pct > 0.70
            ):
                label = "Chop"
        labels.append(
            {
                "regime_id": r,
                "label": label,
                "price_slope": ps,
                "ma_bias": mb,
                "volatility": vol,
                "bb_width_avg": bb,
                "pct_inside_1std": pct,
                "avg_dd_depth": dd,
            }
        )

    labels_df = pd.DataFrame(labels)
    labels_path = audit_dir / f"regime_labels_{tag}.csv"
    labels_df.to_csv(labels_path, index=False)

    print("[AUDIT] Top drivers:")
    for r in range(K):
        label = labels_df.loc[labels_df["regime_id"] == r, "label"].item()
        delta = centroids_unscaled[r] - global_mean
        top_idx = np.argsort(np.abs(delta))[::-1][:3]
        parts = []
        for idx in top_idx:
            sign = "+" if delta[idx] >= 0 else "-"
            parts.append(f"{feat[idx]}({sign})")
        print(f"  R{r}({label}?): " + ", ".join(parts))

    bp_df = pd.DataFrame(block_plan)
    bp_df.insert(0, "block_id", np.arange(1, len(bp_df) + 1))
    assign_dates = assignments.merge(
        bp_df[["block_id", "train_start", "train_end"]], on="block_id", how="left"
    )
    assign_dates = assign_dates.merge(
        labels_df[["regime_id", "label"]], on="regime_id", how="left"
    )
    assign_dates = assign_dates[
        ["block_id", "regime_id", "label", "train_start", "train_end"]
    ]
    assign_path = audit_dir / f"assignments_with_dates_{tag}.csv"
    assign_dates.to_csv(assign_path, index=False)
