from __future__ import annotations

"""Position-based buy evaluation."""

from typing import Dict, Any

from systems.scripts.strategy_pressure import pressure_buy_signal
from systems.utils.addlog import addlog


def evaluate_buy(
    ctx: Dict[str, Any],
    t: int,
    series,
    *,
    window_name: str,
    cfg: Dict[str, Any],
    runtime_state: Dict[str, Any],
):
    """Return sizing and metadata for a buy signal in ``window_name``.

    Parameters
    ----------
    ctx:
        Context dictionary containing at least a ``ledger`` instance.
    t:
        Current candle index within ``series``.
    series:
        Candle DataFrame with at least ``close``, ``low`` and ``high`` columns.
    window_name:
        Name of the window configuration under evaluation.
    cfg:
        Strategy configuration for ``window_name``.
    runtime_state:
        Mutable dictionary carrying ``capital`` and ``buy_unlock_p`` mapping.
    """

    verbose = runtime_state.get("verbose", 0)

    candle = series.iloc[t].to_dict()

    if not pressure_buy_signal(candle, runtime_state):
        return None

    price = float(candle.get("close", 0.0))

    capital = float(runtime_state.get("capital", 0.0))
    limits = runtime_state.get("limits", {})
    max_sz = float(limits.get("max_note_usdt", capital))
    min_sz = float(limits.get("min_note_size", 0.0))
    size_usd = min(capital, max_sz)
    if size_usd < min_sz or size_usd <= 0:
        addlog(
            f"[SKIP] size=${size_usd:.2f} < min=${min_sz:.2f}",
            verbose_int=2,
            verbose_state=verbose,
        )
        return None

    addlog(
        f"[PRESSURE_BUY] price=${price:.4f} size=${size_usd:.2f}",
        verbose_int=1,
        verbose_state=verbose,
    )

    result = {
        "action": "BUY",
        "price": price,
        "size_usd": size_usd,
        "window_name": window_name,
    }
    if "timestamp" in series.columns:
        result["created_ts"] = int(candle.get("timestamp", 0))
    result["created_idx"] = t
    return result
