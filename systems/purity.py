from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd, json
from systems.paths import raw_parquet, temp_audit_dir, temp_blocks_dir, BRAINS_DIR
from systems.brain import RegimeBrain
from systems.features import extract_features, ALL_FEATURES


def _duration_to_candles(dur: str) -> int:
    # 1h candles only; supports 'w','d','m' minimally
    dur = dur.strip().lower()
    if dur.endswith("w"): return int(float(dur[:-1]) * 7 * 24)
    if dur.endswith("d"): return int(float(dur[:-1]) * 24)
    if dur.endswith("m"): return int(float(dur[:-1]) * 30 * 24)
    if dur.endswith("h"): return int(float(dur[:-1]))
    raise ValueError(f"Unsupported duration: {dur}")


def _load_blocks(run_id: str, tag: str) -> pd.DataFrame:
    bp = max((temp_blocks_dir(run_id)).glob(f"block_plan_{tag}.json"))
    plan = json.loads(Path(bp).read_text())
    # Plan may be a dict with "blocks" or a plain list
    blocks = plan.get("blocks") if isinstance(plan, dict) else plan
    return pd.DataFrame(blocks)


def _classify_window(brain: RegimeBrain, df_win: pd.DataFrame) -> np.ndarray:
    # extract raw features, align to brain feature order, scale, return probs via nearest centroid distances → softmax-like?
    # For purity we only need p (brain transitions are for next); here we approximate class probs by a softmin over distance.
    import numpy as np
    feats_raw = extract_features(df_win)   # order = ALL_FEATURES
    feat_order = brain._b["features"]
    idx = [ALL_FEATURES.index(f) for f in feat_order]
    x = feats_raw[idx].astype(float)

    mean = np.array(brain._b["scaler"]["mean"], float)
    std  = np.array(brain._b["scaler"]["std"], float)
    std  = np.maximum(std, float(brain._b["scaler"].get("std_floor", 1e-6)))
    z = (x - mean) / std

    C = np.asarray(brain._b["centroids"], float)  # (K,F)
    d2 = ((C - z[None, :])**2).sum(axis=1)        # squared distances
    # Convert to pseudo-probabilities via softmin over distances
    # temp fixed at 1.0 for purity; can expose later if needed
    w = np.exp(-d2)
    p = w / w.sum()
    return p


def compute_purity(tag: str, run_id: str, tau: float, win_dur: str, stride: int) -> Path:
    # Load brain + candles + blocks
    brain_path = BRAINS_DIR / f"brain_{tag}.json"
    brain = RegimeBrain.from_file(brain_path)

    df = pd.read_parquet(raw_parquet(tag)).sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    blocks = _load_blocks(run_id, tag)
    win = _duration_to_candles(win_dur)

    rows = []
    for _, b in blocks.iterrows():
        s = int(b["train_index_start"]); e = int(b["train_index_end"])
        train = df.iloc[s:e+1]
        if len(train) < win + 1:
            rows.append({
                "block_id": int(b["block_id"]) if "block_id" in b else int(len(rows)+1),
                "samples": 0,
                "mean_p0": 0.0, "mean_p1": 0.0, "mean_p2": 0.0,
                "purity0": 0.0, "purity1": 0.0, "purity2": 0.0,
                "pure_regime": -1, "purity": 0.0,
                "win_candles": win, "stride": stride
            })
            continue

        ps = []
        # slide inside train slice
        for i in range(0, len(train) - win + 1, stride):
            sub = train.iloc[i:i+win]
            p = _classify_window(brain, sub)
            ps.append(p)
        P = np.array(ps) if ps else np.zeros((0, len(brain._b["centroids"])))
        if P.size == 0:
            mean_p = np.zeros((len(brain._b["centroids"]),))
            purity = np.zeros_like(mean_p)
        else:
            mean_p = P.mean(axis=0)
            # argmax frequency per regime
            arg = P.argmax(axis=1)
            k = mean_p.size
            purity = np.array([(arg == r).mean() for r in range(k)])

        pure_regime = int(np.argmax(purity)) if P.size else -1
        row = {
            "block_id": int(b.get("block_id", len(rows)+1)),
            "samples": int(len(ps)),
            **{f"mean_p{r}": float(mean_p[r]) for r in range(mean_p.size)},
            **{f"purity{r}": float(purity[r]) for r in range(purity.size)},
            "pure_regime": pure_regime,
            "purity": float(purity[pure_regime]) if pure_regime >= 0 else 0.0,
            "win_candles": int(win),
            "stride": int(stride)
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    out_dir = temp_audit_dir(run_id) / "purity"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"purity_{tag}.csv"
    out.to_csv(out_path, index=False)

    # Print counts at tau
    if "purity" in out.columns:
        k = mean_p.size if rows else 0
        counts = {}
        for r in range(k):
            counts[r] = int(((out["pure_regime"] == r) & (out["purity"] >= tau)).sum())
        print(f"[PURITY] τ={tau} | pure blocks:", counts)
    else:
        print("[PURITY] No rows computed.")
    return out_path
