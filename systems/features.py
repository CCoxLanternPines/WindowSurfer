from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "mean_return",
    "volatility",
    "skew_return",
    "kurt_return",
    "max_drawdown",
    "up_day_ratio",
    "avg_volume",
    "vol_slope",
    "price_slope",
    "price_range",
]


def _safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


def extract_features(df: pd.DataFrame) -> np.ndarray:
    """Compute feature vector for a single training window.

    Parameters
    ----------
    df : pd.DataFrame
        Candle data slice for the training window.
    """
    close = df["close"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)

    log_prices = np.log(close)
    log_returns = np.diff(log_prices)

    mean_return = log_returns.mean() if len(log_returns) else 0.0
    volatility = log_returns.std() if len(log_returns) else 0.0

    centered = log_returns - mean_return
    skew_return = (
        _safe_div(np.mean(centered ** 3), volatility ** 3)
        if len(log_returns)
        else 0.0
    )
    kurt_return = (
        _safe_div(np.mean(centered ** 4), volatility ** 4) - 3.0
        if len(log_returns)
        else 0.0
    )

    running_max = np.maximum.accumulate(close)
    drawdowns = close / running_max - 1.0
    max_drawdown = -drawdowns.min() if len(drawdowns) else 0.0

    up_day_ratio = np.mean(log_returns > 0) if len(log_returns) else 0.0

    avg_volume = volume.mean() if len(volume) else 0.0

    x = np.arange(len(volume))
    x_mean = x.mean() if len(x) else 0.0
    vol_slope = (
        _safe_div(np.sum((x - x_mean) * (volume - volume.mean())), np.sum((x - x_mean) ** 2))
        if len(volume) > 1
        else 0.0
    )

    price_slope_raw = (
        _safe_div(np.sum((x - x_mean) * (close - close.mean())), np.sum((x - x_mean) ** 2))
        if len(close) > 1
        else 0.0
    )
    price_slope = _safe_div(price_slope_raw, close.mean()) if len(close) else 0.0

    price_range = _safe_div(close.max() - close.min(), close.mean()) if len(close) else 0.0

    return np.array([
        mean_return,
        volatility,
        skew_return,
        kurt_return,
        max_drawdown,
        up_day_ratio,
        avg_volume,
        vol_slope,
        price_slope,
        price_range,
    ])


def extract_all_features(candles: pd.DataFrame, blocks: list[dict]) -> pd.DataFrame:
    """Extract features for all blocks.

    Parameters
    ----------
    candles : pd.DataFrame
        Full candle DataFrame.
    blocks : list[dict]
        Block definitions from ``plan_blocks``.
    """
    rows = []
    for idx, block in enumerate(blocks, start=1):
        start = block["train_index_start"]
        end = block["train_index_end"] + 1
        window = candles.iloc[start:end]
        features = extract_features(window)
        rows.append([idx, *features])

    columns = ["block_id", *FEATURE_NAMES]
    return pd.DataFrame(rows, columns=columns)


def scale_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, list]]:
    """Standardize features and return scaled DataFrame and metadata."""
    feature_matrix = df[FEATURE_NAMES].to_numpy(dtype=float)
    mean = feature_matrix.mean(axis=0)
    std = feature_matrix.std(axis=0)
    std[std == 0] = 1
    scaled = (feature_matrix - mean) / std
    scaled_df = pd.DataFrame(scaled, columns=FEATURE_NAMES)
    scaled_df.insert(0, "block_id", df["block_id"])
    meta = {"mean": mean.tolist(), "std": std.tolist(), "features": FEATURE_NAMES}
    return scaled_df, meta


def save_features(df: pd.DataFrame, tag: str, timestamp: str) -> Dict[str, Path]:
    """Save unscaled and scaled features along with metadata."""
    features_dir = Path("features")
    features_dir.mkdir(exist_ok=True)

    raw_path = features_dir / f"features_{tag}_{timestamp}.parquet"
    df.to_parquet(raw_path, index=False)

    scaled_df, meta = scale_features(df)
    scaled_path = features_dir / f"features_scaled_{tag}_{timestamp}.parquet"
    scaled_df.to_parquet(scaled_path, index=False)

    meta_path = features_dir / f"features_meta_{tag}_{timestamp}.json"
    with meta_path.open("w") as fh:
        json.dump(meta, fh, indent=2)

    return {"raw": raw_path, "scaled": scaled_path, "meta": meta_path}
