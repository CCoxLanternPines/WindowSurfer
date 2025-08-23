from __future__ import annotations

"""Helpers for computing trend/volatility regimes."""

from pathlib import Path
import json
from typing import Dict

import numpy as np
import pandas as pd


def load_regime_settings() -> Dict[str, float]:
    """Load regime parameters from settings/config.json."""
    cfg_path = Path(__file__).resolve().parents[2] / "settings" / "config.json"
    try:
        with cfg_path.open("r", encoding="utf-8") as fh:
            cfg = json.load(fh).get("regime", {})
    except Exception:
        cfg = {}
    return {
        "slope_window": int(cfg.get("slope_window", 20)),
        "vol_window": int(cfg.get("vol_window", 20)),
        "slope_thresh": float(cfg.get("slope_thresh", 0.001)),
    }


def compute_regimes(
    df: pd.DataFrame,
    *,
    slope_window: int = 20,
    vol_window: int = 20,
    slope_thresh: float = 0.001,
) -> pd.DataFrame:
    """Return DataFrame with trend/vol regime tags aligned to ``df``."""
    closes = df["close"].to_numpy()
    trend_tags = []
    for i in range(len(df)):
        if i < slope_window:
            trend_tags.append("unknown")
            continue
        y = closes[i - slope_window + 1 : i + 1]
        x = np.arange(len(y))
        slope = float(np.polyfit(x, y, 1)[0]) if len(y) > 1 else 0.0
        price = y[-1] if y[-1] != 0 else 1.0
        slope_norm = slope / price
        if slope_norm > slope_thresh:
            trend_tags.append("uptrend")
        elif slope_norm < -slope_thresh:
            trend_tags.append("downtrend")
        else:
            trend_tags.append("chop")

    returns = pd.Series(closes).pct_change()
    vol = returns.rolling(vol_window).std()
    vol_med = vol.rolling(vol_window).median()
    vol_tags = []
    for i in range(len(df)):
        if i < vol_window or pd.isna(vol.iloc[i]) or pd.isna(vol_med.iloc[i]):
            vol_tags.append("unknown")
            continue
        vol_tags.append("high-vol" if vol.iloc[i] > vol_med.iloc[i] else "low-vol")

    return pd.DataFrame({"trend": trend_tags, "vol": vol_tags}, index=df.index)


def tag_regime_at(
    df: pd.DataFrame,
    idx: int = -1,
    *,
    slope_window: int = 20,
    vol_window: int = 20,
    slope_thresh: float = 0.001,
) -> Dict[str, str]:
    """Return regime tags for a single candle at ``idx``."""
    if idx < 0:
        idx = len(df) + idx
    sub = df.iloc[: idx + 1]
    regime_df = compute_regimes(
        sub, slope_window=slope_window, vol_window=vol_window, slope_thresh=slope_thresh
    )
    if len(regime_df) == 0:
        return {"trend": "unknown", "vol": "unknown"}
    row = regime_df.iloc[-1]
    return {"trend": str(row["trend"]), "vol": str(row["vol"])}
