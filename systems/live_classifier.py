from __future__ import annotations
from pathlib import Path
import json, numpy as np, pandas as pd
from datetime import datetime, timezone
from systems.brain import RegimeBrain
from systems.paths import raw_parquet


def _parse_ts(ts: str) -> pd.Timestamp:
    return pd.Timestamp(ts).tz_convert("UTC") if "Z" not in ts else pd.Timestamp(ts)


def _load_candles(tag: str, csv_path: str | None) -> pd.DataFrame:
    if csv_path:
        df = pd.read_csv(csv_path)
    else:
        df = pd.read_parquet(raw_parquet(tag))
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def _window_for(df: pd.DataFrame, train_candles: int, at_ts: str | None) -> pd.DataFrame:
    if at_ts:
        t = _parse_ts(at_ts)
        df = df[df["timestamp"] <= t]
    return df.iloc[-train_candles:]


def _compute_features_block(df: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    from systems.features import extract_features
    feats = extract_features(df)
    from systems.features import ALL_FEATURES
    idx = [ALL_FEATURES.index(f) for f in feature_names]
    return feats[idx]


def classify(tag: str, train_candles: int, at_ts: str | None = None, csv_path: str | None = None) -> dict:
    brain_path = Path("data/brains") / f"brain_{tag}.json"
    b = RegimeBrain.from_file(brain_path)
    feat_order = b._b["features"]
    mean = np.array(b._b["scaler"]["mean"], float)
    std = np.array(b._b["scaler"]["std"], float)
    std_floor = float(b._b["scaler"].get("std_floor", 1e-6))
    std = np.maximum(std, std_floor)

    df = _load_candles(tag, csv_path)
    win = _window_for(df, train_candles, at_ts)

    raw = _compute_features_block(win, feat_order)
    x_scaled = (raw - mean) / std

    rid = b.classify_scaled(x_scaled)
    probs_next = b.next_probs(rid)
    return {
        "regime_id": int(rid),
        "probs_next": [float(p) for p in probs_next],
        "features_used": len(feat_order),
    }
