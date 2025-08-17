from __future__ import annotations

"""Buy evaluation driven by predictive pressures."""

from typing import Any, Dict

from systems.scripts.trend_predict import (
    compute_window_features,
    rule_predict,
    update_pressures,
    classify_slope,
)
from systems.utils.addlog import addlog


def evaluate_buy(
    ctx: Dict[str, Any],
    t: int,
    series,
    *,
    cfg: Dict[str, Any],
    runtime_state: Dict[str, Any],
):
    """Return sizing and metadata for a buy signal."""

    window_name = "strategy"
    strategy = cfg or runtime_state.get("strategy", {})
    window_size = int(strategy.get("window_size", 0))
    step = int(strategy.get("window_step", 1))
    start = t + 1 - window_size
    if start < 0 or start % step != 0:
        return False

    verbose = runtime_state.get("verbose", 0)

    features = compute_window_features(series, start, window_size)
    last = runtime_state.setdefault("last_features", {}).get(window_name)
    if last is not None:
        pred = rule_predict(last, strategy)
        slope_cls = classify_slope(last.get("slope", 0.0), strategy.get("flat_band_deg", 10.0))
        update_pressures(runtime_state, window_name, pred, slope_cls, strategy)
    runtime_state["last_features"][window_name] = features

    pressures = runtime_state.setdefault("pressures", {"buy": {}, "sell": {}})
    buy_p = pressures["buy"].get(window_name, 0.0)
    max_p = strategy.get("max_pressure", 1.0)
    if buy_p < strategy.get("buy_trigger", 0.0):
        return False

    fraction = buy_p / max_p if max_p else 0.0
    capital = runtime_state.get("capital", 0.0)
    limits = runtime_state.get("limits", {})
    max_sz = float(limits.get("max_note_usdt", capital))
    min_sz = float(limits.get("min_note_size", 0.0))

    raw = capital * fraction
    size_usd = min(raw, capital, max_sz)
    if size_usd != raw:
        addlog(
            f"[CLAMP] size=${raw:.2f} → ${size_usd:.2f} (cap=${capital:.2f}, max=${max_sz:.2f})",
            verbose_int=2,
            verbose_state=verbose,
        )
    if size_usd < min_sz:
        addlog(
            f"[SKIP][{window_name} {window_size}] size=${size_usd:.2f} < min=${min_sz:.2f}",
            verbose_int=2,
            verbose_state=verbose,
        )
        return False

    addlog(
        f"[BUY][{window_name} {window_size}] pressure={buy_p:.1f}/{max_p:.1f} spend=${size_usd:.2f}",
        verbose_int=1,
        verbose_state=verbose,
    )

    pressures["buy"][window_name] = 0.0

    result = {
        "size_usd": size_usd,
        "window_name": window_name,
        "window_size": window_size,
        "p_buy": fraction,
        "unlock_p": None,
    }
    candle = series.iloc[t]
    if "timestamp" in series.columns:
        result["created_ts"] = int(candle.get("timestamp"))
    result["created_idx"] = t
    return result
