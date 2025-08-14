from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from tools.utils.signal_loader import load_signal_modules
from tools.utils.context_features import compute_context_features
from tools.regime_detector import kmeans_regimes


# ---------------------------------------------------------------------------
# Helpers


def load_candles(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
    df.columns = [c.strip().lower() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df = df.dropna(subset=["close"]).sort_values("timestamp").reset_index(drop=True)
    return df


def build_labels(df: pd.DataFrame, horizons: List[int]) -> Dict[int, pd.Series]:
    labels: Dict[int, pd.Series] = {}
    for h in horizons:
        diff = df["close"].shift(-h) - df["close"]
        lab = np.sign(diff)
        lab[lab == 0] = np.nan
        labels[h] = lab
    return labels


def parse_folds(df: pd.DataFrame, span: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    if span.lower().endswith("m"):
        months = int(span[:-1])
        start = df["timestamp"].iloc[0]
        while True:
            train_end = start + pd.DateOffset(months=months)
            test_end = train_end + pd.DateOffset(months=months)
            train_idx = df["timestamp"] < train_end
            test_idx = (df["timestamp"] >= train_end) & (df["timestamp"] < test_end)
            if test_idx.sum() == 0:
                break
            folds.append((train_idx.values, test_idx.values))
            start = train_end
    else:
        step = int(span)
        n = len(df)
        for start in range(0, n, step):
            train_end = start + step
            test_end = train_end + step
            if train_end >= n:
                break
            train_idx = np.arange(n) < train_end
            test_idx = (np.arange(n) >= train_end) & (np.arange(n) < test_end)
            folds.append((train_idx, test_idx))
    return folds


def ece_score(prob: np.ndarray, label01: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.digitize(prob, edges[1:-1], right=True)
    ece = 0.0
    n = len(prob)
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        acc = label01[m].mean()
        conf = prob[m].mean()
        ece += (m.sum() / n) * abs(acc - conf)
    return float(ece)


# ---------------------------------------------------------------------------
# Main logic


def main() -> None:
    parser = argparse.ArgumentParser(description="Horizon profiler")
    parser.add_argument("--csv", required=True, help="Candle CSV path")
    parser.add_argument("--horizons", default="1,2,3,4,5,6,7,8,9,10,11,12")
    parser.add_argument("--fold-span", default="6M")
    parser.add_argument("--bins", type=int, default=3)
    parser.add_argument("--min-fires", type=int, default=50)
    parser.add_argument("--out", default="tools/out/horizon_profiler")
    parser.add_argument("--kmeans", type=int, default=0)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = base_dir / csv_path
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = base_dir / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    horizons = [int(x) for x in args.horizons.split(",") if x]

    df = load_candles(csv_path)
    labels = build_labels(df, horizons)
    ctx = compute_context_features(df)

    modules = load_signal_modules(base_dir / "signal_modules")
    if not modules:
        print("No signal modules found")
        return

    # Prepare dataset
    data = pd.DataFrame({"timestamp": df["timestamp"], "i": np.arange(len(df)), "close": df["close"]})
    data = pd.concat([data, ctx], axis=1)
    for m in modules:
        name = m["name"]
        raw_vals: List[float] = []
        for i in range(len(df)):
            if i < m.get("lookback", 0):
                raw_vals.append(np.nan)
                continue
            raw_vals.append(float(m["calculate"](df, i)))
        data[f"raw_{name}"] = raw_vals
    for h in horizons:
        data[f"y_{h}"] = labels[h]
    # Probability columns
    for m in modules:
        for h in horizons:
            data[f"P_{m['name']}_h{h}"] = np.nan

    folds = parse_folds(df, args.fold_span)
    test_mask = np.zeros(len(df), dtype=bool)

    calib_tables: Dict[Tuple[str, int], List[pd.DataFrame]] = {}

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        test_mask |= test_idx
        for h in horizons:
            y = labels[h]
            y_train = y[train_idx]
            prior = (y_train == 1).mean()
            if np.isnan(prior):
                prior = 0.5
            for m in modules:
                name = m["name"]
                raw = data[f"raw_{name}"]
                x_train = raw[train_idx]
                mask = (~x_train.isna()) & (~y_train.isna())
                x_train = x_train[mask]
                y_sub = y_train[mask]
                if len(x_train) == 0:
                    continue
                is_categorical = np.isin(x_train.unique(), [-1, 0, 1]).all()
                prob_col = f"P_{name}_h{h}"
                records = []
                if is_categorical:
                    for val in [-1, 0, 1]:
                        mask_val = x_train == val
                        fires = mask_val.sum()
                        wins = (y_sub[mask_val] == 1).sum()
                        prob = wins / fires if fires >= args.min_fires and fires > 0 else prior
                        records.append({
                            "fold": fold_idx,
                            "value": val,
                            "fires": int(fires),
                            "wins": int(wins),
                            "prob": float(prob),
                        })
                    tbl = {r["value"]: r for r in records}
                    x_test = raw[test_idx]
                    for idx in np.where(test_idx)[0]:
                        val = raw.iloc[idx]
                        rec = tbl.get(val)
                        if rec and rec["fires"] >= args.min_fires:
                            data.at[idx, prob_col] = rec["prob"]
                        else:
                            data.at[idx, prob_col] = prior
                else:
                    q = np.linspace(0, 1, args.bins + 1)
                    edges = np.unique(np.quantile(x_train, q))
                    if len(edges) <= 1:
                        edges = np.array([x_train.min(), x_train.max()])
                    bins = np.digitize(x_train, edges[1:-1], right=True)
                    for b in range(len(edges) - 1):
                        mask_bin = bins == b
                        fires = mask_bin.sum()
                        wins = (y_sub[mask_bin] == 1).sum()
                        prob = wins / fires if fires >= args.min_fires and fires > 0 else prior
                        records.append({
                            "fold": fold_idx,
                            "bin": b,
                            "low": edges[b],
                            "high": edges[b + 1],
                            "fires": int(fires),
                            "wins": int(wins),
                            "prob": float(prob),
                        })
                    x_test = raw[test_idx]
                    test_bins = np.digitize(x_test, edges[1:-1], right=True)
                    for idx, b in zip(np.where(test_idx)[0], test_bins):
                        recs = [r for r in records if r.get("bin") == b]
                        rec = recs[0] if recs else None
                        if rec and rec["fires"] >= args.min_fires:
                            data.at[idx, prob_col] = rec["prob"]
                        else:
                            data.at[idx, prob_col] = prior
                calib_tables.setdefault((name, h), []).append(pd.DataFrame(records))

    # Save dataset and probability matrices
    data.to_parquet(out_dir / "dataset.parquet", index=False)
    for m in modules:
        cols = [f"P_{m['name']}_h{h}" for h in horizons]
        probs = data[cols].copy()
        probs.columns = [f"h{h}" for h in horizons]
        probs.to_parquet(out_dir / f"probs__brain={m['name']}.parquet", index=False)
    for (name, h), tables in calib_tables.items():
        tbl = pd.concat(tables, ignore_index=True)
        tbl.to_csv(out_dir / f"calibration__brain={name}__h={h}.csv", index=False)

    # Leaderboard overall
    records = []
    n_total = test_mask.sum()
    for m in modules:
        name = m["name"]
        for h in horizons:
            prob_col = f"P_{name}_h{h}"
            label_col = f"y_{h}"
            raw_col = f"raw_{name}"
            mask = test_mask & data[prob_col].notna() & data[label_col].notna()
            if mask.sum() == 0:
                continue
            fires = ((data[raw_col] != 0) & mask).sum()
            coverage = fires / mask.sum() if mask.sum() else 0.0
            label = data[label_col][mask]
            prob = data[prob_col][mask]
            label01 = (label + 1) / 2
            brier = float(((prob - label01) ** 2).mean())
            pred = np.where(prob >= 0.5, 1, -1)
            hit = float((pred == label).mean())
            ece = ece_score(prob.values, label01.values)
            records.append({
                "brain": name,
                "h": h,
                "fires": int(fires),
                "hit_rate": hit,
                "coverage": coverage,
                "brier": brier,
                "ece": ece,
            })
    df_lead = pd.DataFrame(records)
    df_lead.to_csv(out_dir / "leaderboard_overall.csv", index=False)
    if not df_lead.empty:
        df_print = df_lead.sort_values(["brier", "hit_rate"], ascending=[True, False])
        print(df_print.to_string(index=False))

    # Regime detection
    if args.kmeans > 0:
        prob_cols = [f"P_{m['name']}_h{h}" for m in modules for h in horizons]
        feature_cols = prob_cols + [c for c in data.columns if c.startswith("ctx_")]
        features = data[feature_cols]
        regimes = kmeans_regimes(features, args.kmeans)
        if regimes is not None:
            data = data.join(regimes)
            data[["timestamp", "regime_label"]].to_parquet(out_dir / "regimes.parquet", index=False)
            # Per-regime leaderboards
            reg_records = []
            for reg, idxs in data.dropna(subset=["regime_label"]).groupby("regime_label").groups.items():
                sub = data.loc[idxs]
                for m in modules:
                    name = m["name"]
                    for h in horizons:
                        prob_col = f"P_{name}_h{h}"
                        label_col = f"y_{h}"
                        raw_col = f"raw_{name}"
                        mask = sub[prob_col].notna() & sub[label_col].notna()
                        if mask.sum() == 0:
                            continue
                        fires = ((sub[raw_col] != 0) & mask).sum()
                        label = sub[label_col][mask]
                        prob = sub[prob_col][mask]
                        label01 = (label + 1) / 2
                        brier = float(((prob - label01) ** 2).mean())
                        pred = np.where(prob >= 0.5, 1, -1)
                        hit = float((pred == label).mean())
                        ece = ece_score(prob.values, label01.values)
                        reg_records.append({
                            "regime": int(reg),
                            "brain": name,
                            "h": h,
                            "fires": int(fires),
                            "hit_rate": hit,
                            "brier": brier,
                            "ece": ece,
                        })
            if reg_records:
                df_reg = pd.DataFrame(reg_records)
                df_reg.to_csv(out_dir / "leaderboard_by_regime.csv", index=False)
                # Print top 5 per regime by hit rate
                for reg in sorted(df_reg["regime"].unique()):
                    sub = df_reg[df_reg["regime"] == reg]
                    sub = sub.sort_values("hit_rate", ascending=False).head(5)
                    print(f"Regime {reg}:")
                    print(sub.to_string(index=False))


if __name__ == "__main__":
    main()
