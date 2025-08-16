from __future__ import annotations

"""Causal regime detector for simulations and live mode.

Computes a regime score in ``[-1,1]`` and a discrete label ``{-1,0,1}``
using only data available up to ``t-1``.
"""

from dataclasses import dataclass
from typing import Iterable
import numpy as np
import pandas as pd


@dataclass
class Params:
    fast: int = 24
    slow: int = 96
    slope_win: int = 48
    vol_win: int = 96
    w1: float = 0.6
    w2: float = 0.4
    bull_th: float = 0.25
    bear_th: float = -0.25
    min_bars: int = 12


DEFAULT_PARAMS = Params()


def _rolling_slope(arr: Iterable[float]) -> float:
    x = np.arange(len(arr))
    y = np.asarray(arr)
    if len(y) < 2:
        return 0.0
    beta, _ = np.polyfit(x, y, 1)
    return float(beta)


def compute_regime_point(
    series: pd.DataFrame,
    t: int,
    *,
    p: Params | None = None,
    prev_label: int = 0,
    prev_switch_t: int = -10**9,
) -> dict:
    """Return causal regime ``score`` and ``label`` for index ``t``.

    ``series`` must contain a ``close`` column. Only data ``<= t-1`` is
    used via ``shift(1)``. ``prev_label`` and ``prev_switch_t`` implement
    hysteresis with a minimum duration guard.
    """

    p = p or DEFAULT_PARAMS
    logp = np.log(pd.to_numeric(series["close"], errors="coerce"))
    ema_fast = logp.ewm(span=p.fast, adjust=False).mean().shift(1)
    ema_slow = logp.ewm(span=p.slow, adjust=False).mean().shift(1)

    diff = (ema_fast - ema_slow).iloc[t]
    slow = ema_slow.iloc[t]

    slope = (
        ema_slow.rolling(p.slope_win)
        .apply(_rolling_slope, raw=False)
        .shift(1)
        .iloc[t]
    )
    vol = logp.diff().rolling(p.vol_win).std().shift(1).iloc[t]
    if not np.isfinite(vol) or vol <= 0:
        vol = 1e-8

    diff_score = np.tanh(np.clip(diff / vol, -3, 3) / 2)
    trend_score = np.tanh(np.clip(slope / vol, -3, 3) / 2)
    score = np.clip(p.w1 * trend_score + p.w2 * diff_score, -1.0, 1.0)

    label = prev_label
    now = t
    if score >= p.bull_th and (
        prev_label != 1 and (score >= 0.5 or now - prev_switch_t >= p.min_bars)
    ):
        label = 1
        prev_switch_t = now
    elif score <= p.bear_th and (
        prev_label != -1 and (score <= -0.5 or now - prev_switch_t >= p.min_bars)
    ):
        label = -1
        prev_switch_t = now

    if __debug__ and t > 1 and getattr(p, "_assert", True) and np.random.randint(0, 5000) == 0:
        sub = series.iloc[: t].copy()
        p2 = Params(**vars(p))
        setattr(p2, "_assert", False)
        compute_regime_point(
            sub,
            t - 1,
            p=p2,
            prev_label=prev_label,
            prev_switch_t=prev_switch_t,
        )

    return {"score": float(score), "label": int(label)}
