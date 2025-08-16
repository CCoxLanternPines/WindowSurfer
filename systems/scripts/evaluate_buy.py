from __future__ import annotations

"""Pressure-based buy evaluator."""

from typing import Any, Dict
import pandas as pd

from systems.utils.addlog import addlog


# --- tuning constants -----------------------------------------------------

PRESSURE_WINDOWS = [
    ("3m", 0.20),
    ("1m", 0.20),
    ("2w", 0.15),
    ("1w", 0.15),
    ("3d", 0.15),
    ("1d", 0.15),
]

BUY_CD_MIN = 2  # bars
BUY_CD_MAX = 48  # bars
BASE_BUY_FRACTION = 0.10  # size = capital * this


def _clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def _span_to_bars(series: pd.DataFrame, span: str) -> int:
    if not span:
        return 1
    try:
        value = int(span[:-1])
    except ValueError:
        return 1
    unit = span[-1].lower()
    if "timestamp" in series.columns and len(series) >= 2:
        step = int(series.iloc[1]["timestamp"]) - int(series.iloc[0]["timestamp"])
        if step <= 0:
            step = 3600
    else:
        step = 3600
    if unit == "m":
        factor = 30 * 24 * 3600
    elif unit == "w":
        factor = 7 * 24 * 3600
    elif unit == "d":
        factor = 24 * 3600
    elif unit == "h":
        factor = 3600
    else:
        factor = 0
    bars = int(round((value * factor) / step))
    return max(1, bars)


def _window_scores(series: pd.DataFrame, t: int) -> Dict[str, Dict[str, float]]:
    scores: Dict[str, Dict[str, float]] = {}
    close_now = float(series.iloc[t]["close"])
    for span, _ in PRESSURE_WINDOWS:
        bars = _span_to_bars(series, span)
        start = max(0, t - bars + 1)
        window = series.iloc[start : t + 1]
        low = float(window["close"].min())
        high = float(window["close"].max())
        width = max(high - low, 1e-9)
        depth = (high - close_now) / width
        height = (close_now - low) / width
        scores[span] = {"depth": depth, "height": height}
    return scores


def _pressure(scores: Dict[str, Dict[str, float]], windows) -> tuple[float, float]:
    wsum = sum(w for _, w in windows) or 1.0
    sell_p = sum(w * scores[s]["height"] for s, w in windows if s in scores) / wsum
    buy_p = sum(w * scores[s]["depth"] for s, w in windows if s in scores) / wsum
    return _clamp01(buy_p), _clamp01(sell_p)


def _cooldown_bars(pressure: float, min_bars: int, max_bars: int) -> int:
    p = _clamp01(pressure)
    return int(round(max_bars - p * (max_bars - min_bars)))


def evaluate_buy(
    ctx: Dict[str, Any],
    t: int,
    series: pd.DataFrame,
    *,
    window_name: str,
    cfg: Dict[str, Any],
    runtime_state: Dict[str, Any],
):
    """Return sizing and metadata for a buy signal in ``window_name``."""

    verbose = runtime_state.get("verbose", 0)

    scores = _window_scores(series, t)
    buy_p, sell_p = _pressure(scores, PRESSURE_WINDOWS)
    cd = _cooldown_bars(buy_p, BUY_CD_MIN, BUY_CD_MAX)

    last_key = f"last_buy_idx::{window_name}"
    last_idx = int(runtime_state.get(last_key, -10**9))
    elapsed = t - last_idx

    decision: Dict[str, Any] | bool = False
    if elapsed >= cd:
        capital = float(runtime_state.get("capital", 0.0))
        size_usd = max(0.0, capital * BASE_BUY_FRACTION)
        decision = {
            "size_usd": size_usd,
            "window_name": window_name,
            "window_size": cfg["window_size"],
            "created_idx": t,
        }
        if "timestamp" in series.columns:
            decision["created_ts"] = int(series.iloc[t]["timestamp"])
        runtime_state[last_key] = t

    addlog(
        f"[{ 'BUY' if decision else 'SKIP' }][{window_name} {cfg['window_size']}] buy_p={buy_p:.3f} cd={cd} elapsed={elapsed}",
        verbose_int=1,
        verbose_state=verbose,
    )

    return decision

