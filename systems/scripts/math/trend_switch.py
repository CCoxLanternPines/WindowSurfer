from __future__ import annotations

"""Tiny trend estimation helpers for switchback detection."""

from typing import List, Tuple
import math
import statistics


def robust_slope_log(prices: List[float]) -> float:
    """Return OLS slope of log-price per bar with median fallback."""
    ln = [math.log(p) for p in prices if p > 0]
    n = len(ln)
    if n < 2:
        return 0.0
    x = list(range(n))
    xm = (n - 1) / 2.0
    ym = statistics.fmean(ln)
    denom = sum((xi - xm) ** 2 for xi in x)
    if denom <= 0:
        denom = 1.0
    slope = sum((xi - xm) * (yi - ym) for xi, yi in zip(x, ln)) / denom
    if not math.isfinite(slope):
        med = statistics.median(ln)
        slope = (ln[-1] - med - (ln[0] - med)) / max(n - 1, 1)
    return slope


def slope_stats(prices: List[float]) -> Tuple[float, float]:
    """Return ``(slope, se)`` for log-price series."""
    ln = [math.log(p) for p in prices if p > 0]
    n = len(ln)
    if n < 2:
        return 0.0, 1e-9
    slope = robust_slope_log(prices)
    x = list(range(n))
    xm = (n - 1) / 2.0
    ym = statistics.fmean(ln)
    intercept = ym - slope * xm
    resid = [yi - (intercept + slope * xi) for xi, yi in zip(x, ln)]
    if n > 2:
        std_res = statistics.stdev(resid)
    else:
        std_res = 0.0
    se = std_res * math.sqrt(12) / (n ** 1.5)
    if se < 1e-9:
        se = 1e-9
    return slope, se


def classify_trend_z(
    slope: float,
    se: float,
    z_hi: float,
    z_lo: float,
    min_abs_slope: float,
) -> str:
    """Return ``'UP'``, ``'DOWN'`` or ``'FLAT'`` based on z-score."""
    if abs(slope) < min_abs_slope or se <= 0:
        return "FLAT"
    z = slope / se
    if z >= z_hi:
        return "UP"
    if z <= z_lo:
        return "DOWN"
    return "FLAT"
