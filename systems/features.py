from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple

import json as _json
import numpy as np
import pandas as pd
import json

from .paths import temp_features_dir


def _feature_sha(features: list[str]) -> str:
    # stable JSON encoding for deterministic hash
    payload = _json.dumps(features, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()

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


def _safe_div(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray | float:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)


def extract_features(df: pd.DataFrame) -> np.ndarray:
    close = np.asarray(df["close"], dtype=float)
    volume = np.asarray(df["volume"], dtype=float)

    close = np.nan_to_num(
        close, nan=np.nanmedian(close) if np.isfinite(np.nanmedian(close)) else 0.0
    )
    volume = np.nan_to_num(volume, nan=0.0)

    n = close.size
    if n == 0:
        return np.zeros(len(ALL_FEATURES), dtype=float)

    eps = 1e-12
    safe_close = np.maximum(close, eps)
    log_returns = np.diff(np.log(safe_close))

    mean_return = float(log_returns.mean()) if log_returns.size else 0.0
    volatility = float(log_returns.std()) if log_returns.size else 0.0

    running_max = np.maximum.accumulate(close)
    dd = close / np.where(running_max == 0.0, 1.0, running_max) - 1.0
    max_drawdown = float(-np.nanmin(dd)) if dd.size else 0.0

    price_range = _safe_div(close.max() - close.min(), close.mean()) if n else 0.0
    up_day_ratio = float(np.mean(log_returns > 0)) if log_returns.size else 0.0
    avg_volume = float(volume.mean()) if volume.size else 0.0

    x = np.arange(n, dtype=float)
    denom = np.sum((x - x.mean()) ** 2)
    price_slope_raw = 0.0 if denom == 0.0 else np.sum((x - x.mean()) * (close - close.mean())) / denom
    price_slope = 0.0 if close.mean() == 0.0 else price_slope_raw / close.mean()

    ma50 = pd.Series(close).rolling(50, min_periods=50).mean().to_numpy()
    valid_ma50 = np.isfinite(ma50)
    if valid_ma50.sum() > 1:
        x_ma = np.arange(n, dtype=float)[valid_ma50]
        y_ma = ma50[valid_ma50]
        denom = np.sum((x_ma - x_ma.mean()) ** 2)
        ma_slope_raw = 0.0 if denom == 0.0 else np.sum((x_ma - x_ma.mean()) * (y_ma - y_ma.mean())) / denom
        ma_slope = 0.0 if y_ma.mean() == 0.0 else ma_slope_raw / y_ma.mean()
    else:
        ma_slope = 0.0

    ma200 = pd.Series(close).rolling(200, min_periods=200).mean().to_numpy()
    valid_ma200 = np.isfinite(ma200)
    ma_bias = float(np.mean(close[valid_ma200] > ma200[valid_ma200])) if valid_ma200.any() else 0.0

    if log_returns.size >= 24:
        rolling_vol = pd.Series(log_returns).rolling(24, min_periods=24).std().to_numpy()
        vol_series = rolling_vol[np.isfinite(rolling_vol)]
        if vol_series.size:
            current_vol = vol_series[-1]
            vol_percentile = float(np.mean(vol_series <= current_vol))
            vol_of_vol = float(np.std(vol_series))
        else:
            vol_percentile = 0.0
            vol_of_vol = 0.0
    else:
        vol_percentile = 0.0
        vol_of_vol = 0.0

    roll_mean = pd.Series(close).rolling(20, min_periods=20).mean().to_numpy()
    roll_std = pd.Series(close).rolling(20, min_periods=20).std().to_numpy()
    bb_width = _safe_div(4.0 * roll_std, roll_mean)
    bb_width_avg = float(np.nanmean(bb_width)) if np.isfinite(bb_width).any() else 0.0

    if n:
        mean_price = close.mean()
        std_price = close.std()
        pct_inside_1std = float(
            np.mean((close >= mean_price - std_price) & (close <= mean_price + std_price))
        )
    else:
        pct_inside_1std = 0.0

    vol_up = volume[1:][log_returns > 0].sum() if volume.size > 1 else 0.0
    vol_down = volume[1:][log_returns <= 0].sum() if volume.size > 1 else 0.0
    up_vol_ratio = float(_safe_div(vol_up, vol_down))

    if (
        log_returns.size
        and np.std(volume[1:]) > 0
        and np.std(log_returns) > 0
    ):
        v = volume[1:].astype(float)
        lr = log_returns.astype(float)
        if np.isfinite(v).all() and np.isfinite(lr).all():
            vol_price_corr = float(np.corrcoef(lr, v)[0, 1])
        else:
            vol_price_corr = 0.0
    else:
        vol_price_corr = 0.0

    dd_lengths: list[int] = []
    dd_depths: list[float] = []
    length = 0
    min_dd = 0.0
    for val in dd:
        if val < 0:
            length += 1
            min_dd = min(min_dd, val)
        elif length:
            dd_lengths.append(length)
            dd_depths.append(-min_dd)
            length = 0
            min_dd = 0.0
    if length:
        dd_lengths.append(length)
        dd_depths.append(-min_dd)
    avg_dd_length = float(np.mean(dd_lengths)) if dd_lengths else 0.0
    avg_dd_depth = float(np.mean(dd_depths)) if dd_depths else 0.0

    feats = np.array(
        [
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
        ],
        dtype=float,
    )
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats


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
    X = df[feature_names].to_numpy(dtype=float)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    eps = 1e-6
    std = np.where(std < eps, eps, std)  # std floor
    Z = (X - mean) / std
    scaled_df = pd.DataFrame(Z, columns=feature_names)
    scaled_df.insert(0, "block_id", df["block_id"].to_numpy())
    meta = {
        "features": feature_names,
        "feature_sha": _feature_sha(feature_names),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "std_floor": eps,
    }
    return scaled_df, meta


def save_features(df: pd.DataFrame, tag: str, run_id: str) -> Dict[str, Path]:
    raw = df.copy()
    nan_cols = [c for c in ALL_FEATURES if raw[c].isna().any()]
    if nan_cols:
        print(f"[FEATURES][WARN] Found NaNs in columns, zero-filling: {nan_cols}")
        raw[nan_cols] = raw[nan_cols].fillna(0.0)

    variances = raw[ALL_FEATURES].std()
    keep = variances[variances >= 1e-6].index.tolist()
    dropped = [f for f in ALL_FEATURES if f not in keep]
    print(f"[FEATURES] Dropped {len(dropped)} features for low variance: {dropped}")
    FEATURE_NAMES[:] = keep

    scaled_df, meta = scale_features(raw, FEATURE_NAMES)

    scaled_arr = scaled_df[FEATURE_NAMES].to_numpy()
    if not np.isfinite(scaled_arr).all():
        bad = np.argwhere(~np.isfinite(scaled_arr))
        raise ValueError(
            f"[FEATURES][FATAL] Non-finite values after scaling at indices: {bad[:5].tolist()} ..."
        )

    features_dir = temp_features_dir(run_id)
    features_dir.mkdir(parents=True, exist_ok=True)

    summary_path = features_dir / f"feature_summary_{tag}.csv"
    summary_df = scaled_df[FEATURE_NAMES].agg(["mean", "std", "min", "max"]).transpose()
    summary_df.to_csv(summary_path)

    features_path = features_dir / f"features_{tag}.parquet"
    scaled_df.to_parquet(features_path, index=False)
    meta_path = features_dir / f"features_meta_{tag}.json"
    with meta_path.open("w") as fh:
        json.dump(meta, fh, indent=2)
    return {"raw": features_path, "meta": meta_path, "summary": summary_path}

