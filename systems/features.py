from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

ALL_FEATURES = [
    "mean_return",
    "volatility",
    "max_drawdown",
    "price_range",
    "up_day_ratio",
    "avg_volume",
    "price_slope",
    "ma_slope",
    "ma_bias",
    "vol_percentile",
    "vol_of_vol",
    "bb_width_avg",
    "pct_inside_1std",
    "up_vol_ratio",
    "vol_price_corr",
    "avg_dd_length",
    "avg_dd_depth",
]
FEATURE_NAMES = ALL_FEATURES.copy()


def _safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


def extract_features(df: pd.DataFrame) -> np.ndarray:
    close = df["close"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)

    log_prices = np.log(close)
    log_returns = np.diff(log_prices)

    mean_return = log_returns.mean() if len(log_returns) else 0.0
    volatility = log_returns.std() if len(log_returns) else 0.0

    running_max = np.maximum.accumulate(close)
    drawdowns = close / running_max - 1.0
    max_drawdown = -drawdowns.min() if len(drawdowns) else 0.0

    price_range = _safe_div(close.max() - close.min(), close.mean()) if len(close) else 0.0
    up_day_ratio = np.mean(log_returns > 0) if len(log_returns) else 0.0
    avg_volume = volume.mean() if len(volume) else 0.0

    x = np.arange(len(close))
    x_mean = x.mean() if len(x) else 0.0
    price_slope_raw = (
        _safe_div(np.sum((x - x_mean) * (close - close.mean())), np.sum((x - x_mean) ** 2))
        if len(close) > 1
        else 0.0
    )
    price_slope = _safe_div(price_slope_raw, close.mean()) if len(close) else 0.0

    ma50 = pd.Series(close).rolling(window=50, min_periods=50).mean().to_numpy()
    valid_ma50 = ~np.isnan(ma50)
    if valid_ma50.sum() > 1:
        x_ma = np.arange(len(ma50))[valid_ma50]
        y_ma = ma50[valid_ma50]
        x_ma_mean = x_ma.mean()
        ma_slope_raw = _safe_div(
            np.sum((x_ma - x_ma_mean) * (y_ma - y_ma.mean())),
            np.sum((x_ma - x_ma_mean) ** 2),
        )
        ma_slope = _safe_div(ma_slope_raw, y_ma.mean())
    else:
        ma_slope = 0.0

    ma200 = pd.Series(close).rolling(window=200, min_periods=200).mean().to_numpy()
    valid_ma200 = ~np.isnan(ma200)
    ma_bias = np.mean(close[valid_ma200] > ma200[valid_ma200]) if valid_ma200.any() else 0.0

    if len(log_returns) >= 24:
        rolling_vol = pd.Series(log_returns).rolling(window=24).std().to_numpy()
        valid_vol = ~np.isnan(rolling_vol)
        vol_series = rolling_vol[valid_vol]
        if len(vol_series):
            current_vol = vol_series[-1]
            vol_percentile = np.mean(vol_series <= current_vol)
            vol_of_vol = vol_series.std()
        else:
            vol_percentile = 0.0
            vol_of_vol = 0.0
    else:
        vol_percentile = 0.0
        vol_of_vol = 0.0

    roll_mean = pd.Series(close).rolling(window=20, min_periods=20).mean().to_numpy()
    roll_std = pd.Series(close).rolling(window=20, min_periods=20).std().to_numpy()
    bb_width = _safe_div(4 * roll_std, roll_mean)
    bb_width_avg = np.nanmean(bb_width) if np.isfinite(bb_width).any() else 0.0

    if len(close):
        mean_price = close.mean()
        std_price = close.std()
        pct_inside_1std = np.mean(
            (close >= mean_price - std_price) & (close <= mean_price + std_price)
        )
    else:
        pct_inside_1std = 0.0

    vol_up = volume[1:][log_returns > 0].sum() if len(volume) > 1 else 0.0
    vol_down = volume[1:][log_returns <= 0].sum() if len(volume) > 1 else 0.0
    up_vol_ratio = _safe_div(vol_up, vol_down)

    if len(log_returns):
        vol_price_corr = (
            np.corrcoef(log_returns, volume[1:])[0, 1]
            if volume[1:].std() and log_returns.std()
            else 0.0
        )
    else:
        vol_price_corr = 0.0

    dd_lengths = []
    dd_depths = []
    length = 0
    min_dd = 0.0
    for dd in drawdowns:
        if dd < 0:
            length += 1
            min_dd = min(min_dd, dd)
        elif length:
            dd_lengths.append(length)
            dd_depths.append(-min_dd)
            length = 0
            min_dd = 0.0
    if length:
        dd_lengths.append(length)
        dd_depths.append(-min_dd)
    avg_dd_length = np.mean(dd_lengths) if dd_lengths else 0.0
    avg_dd_depth = np.mean(dd_depths) if dd_depths else 0.0

    return np.array([
        mean_return,
        volatility,
        max_drawdown,
        price_range,
        up_day_ratio,
        avg_volume,
        price_slope,
        ma_slope,
        ma_bias,
        vol_percentile,
        vol_of_vol,
        bb_width_avg,
        pct_inside_1std,
        up_vol_ratio,
        vol_price_corr,
        avg_dd_length,
        avg_dd_depth,
    ])


def extract_all_features(candles: pd.DataFrame, blocks: list[dict]) -> pd.DataFrame:
    rows = []
    for idx, block in enumerate(blocks, start=1):
        start = block["train_index_start"]
        end = block["train_index_end"] + 1
        window = candles.iloc[start:end]
        features = extract_features(window)
        rows.append([idx, *features])
    columns = ["block_id", *ALL_FEATURES]
    return pd.DataFrame(rows, columns=columns)


def scale_features(
    df: pd.DataFrame, feature_names: list[str] | None = None
) -> Tuple[pd.DataFrame, Dict[str, list]]:
    if feature_names is None:
        feature_names = FEATURE_NAMES
    feature_matrix = df[feature_names].to_numpy(dtype=float)
    mean = feature_matrix.mean(axis=0)
    std = feature_matrix.std(axis=0)
    std[std == 0] = 1
    scaled = (feature_matrix - mean) / std
    scaled_df = pd.DataFrame(scaled, columns=feature_names)
    scaled_df.insert(0, "block_id", df["block_id"].to_numpy())
    meta = {"mean": mean.tolist(), "std": std.tolist(), "features": feature_names}
    return scaled_df, meta


def audit_variance(df: pd.DataFrame, tag: str, timestamp: str) -> Tuple[pd.DataFrame, Path]:
    global FEATURE_NAMES
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    variances = df[FEATURE_NAMES].std()
    variance_path = logs_dir / f"feature_variance_{tag}_{timestamp}.csv"
    variances.to_csv(variance_path, header=["std"])
    keep = variances[variances >= 1e-6].index.tolist()
    dropped = [f for f in FEATURE_NAMES if f not in keep]
    print(f"[FEATURES] Dropped {len(dropped)} features for zero variance: {dropped}")
    FEATURE_NAMES = keep
    return df[["block_id", *keep]], variance_path


def save_features(df: pd.DataFrame, tag: str, timestamp: str) -> Dict[str, Path]:
    features_dir = Path("features")
    features_dir.mkdir(exist_ok=True)
    df_kept, variance_path = audit_variance(df, tag, timestamp)
    scaled_df, meta = scale_features(df_kept)
    features_path = features_dir / f"features_{tag}_{timestamp}.parquet"
    scaled_df.to_parquet(features_path, index=False)
    meta_path = features_dir / f"features_meta_{tag}_{timestamp}.json"
    with meta_path.open("w") as fh:
        json.dump(meta, fh, indent=2)
    return {"raw": features_path, "meta": meta_path, "variance": variance_path}

