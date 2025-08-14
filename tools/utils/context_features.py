from __future__ import annotations

import numpy as np
import pandas as pd


def _rolling_linear(y: pd.Series, window: int) -> pd.DataFrame:
    x = np.arange(window)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()
    slopes = np.full(len(y), np.nan)
    r2s = np.full(len(y), np.nan)
    for i in range(window - 1, len(y)):
        yy = y.iloc[i - window + 1 : i + 1].values
        y_mean = yy.mean()
        cov = ((x - x_mean) * (yy - y_mean)).sum()
        slope = cov / denom
        intercept = y_mean - slope * x_mean
        fitted = slope * x + intercept
        ss_tot = ((yy - y_mean) ** 2).sum()
        ss_res = ((yy - fitted) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
        slopes[i] = slope
        r2s[i] = r2
    return pd.DataFrame(
        {f"ctx_trend_slope_{window}": slopes, f"ctx_trend_r2_{window}": r2s}, index=y.index
    )


def _percent_rank(arr: np.ndarray) -> float:
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.nan
    sorted_arr = np.sort(arr)
    return np.searchsorted(sorted_arr, arr[-1], side="right") / len(sorted_arr)


def compute_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute leak-safe context features for *df*."""

    out = pd.DataFrame(index=df.index)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df.get("volume", pd.Series(index=df.index, dtype=float)).astype(float)

    # Trend features
    for window in (12, 24, 48):
        out = out.join(_rolling_linear(close, window))

    # Volatility features: ATR6, ATR24, ratio
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr6 = tr.rolling(6).mean()
    atr24 = tr.rolling(24).mean()
    out["ctx_atr_6"] = atr6
    out["ctx_atr_24"] = atr24
    out["ctx_atr_ratio_6_24"] = atr6 / atr24

    # Bollinger Band width percentile (20)
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_width = 4 * std20  # (upper - lower)
    out["ctx_bb_width_pct_20"] = bb_width.rolling(20).apply(_percent_rank, raw=True)

    # Volume percentile vs 50-bar EMA
    ema50 = volume.ewm(span=50, adjust=False).mean()
    vol_ratio = volume / ema50
    out["ctx_volume_pct"] = vol_ratio.rolling(50).apply(_percent_rank, raw=True)

    # Distance to rolling high/low
    for w in (20, 50):
        roll_high = close.rolling(w).max()
        roll_low = close.rolling(w).min()
        out[f"ctx_dist_high_{w}"] = (close - roll_high) / roll_high
        out[f"ctx_dist_low_{w}"] = (close - roll_low) / roll_low

    # Close streak length
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i - 1]:
            streak[i] = streak[i - 1] + 1 if streak[i - 1] > 0 else 1
        elif close.iloc[i] < close.iloc[i - 1]:
            streak[i] = streak[i - 1] - 1 if streak[i - 1] < 0 else -1
        else:
            streak[i] = 0
    out["ctx_close_streak"] = streak

    return out
