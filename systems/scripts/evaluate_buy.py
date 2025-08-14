from __future__ import annotations

"""Position-based buy evaluation."""

from typing import Dict, Any

from systems.scripts.window_utils import get_window_bounds, check_buy_conditions
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

    ledger = ctx.get("ledger")
    verbose = runtime_state.get("verbose", 0)

    candle = series.iloc[t]
    win_low, win_high = get_window_bounds(series, t, cfg["window_size"])
    window_data = {"low": win_low, "high": win_high}

    open_notes = []
    if ledger:
        open_notes = [
            n for n in ledger.get_open_notes() if n.get("window_name") == window_name
        ]

    ledger_state = {
        "open_notes": open_notes,
        "buy_unlock_p": runtime_state.setdefault("buy_unlock_p", {}),
        "verbose": verbose,
    }

    ok, meta = check_buy_conditions(
        candle,
        window_data,
        {**cfg, "window_name": window_name},
        ledger_state,
    )
    if not ok:
        return False

    capital = runtime_state.get("capital", 0.0)
    base = cfg.get("investment_fraction", 0.0)
    size_usd = capital * base

    limits = runtime_state.get("limits", {})
    min_sz = float(limits.get("min_note_size", 0.0))
    max_sz = float(limits.get("max_note_usdt", float("inf")))
    raw = size_usd
    size_usd = min(size_usd, capital, max_sz)
    if raw != size_usd:
        addlog(
            f"[CLAMP] size=${raw:.2f} → ${size_usd:.2f} (cap=${capital:.2f}, max=${max_sz:.2f})",
            verbose_int=2,
            verbose_state=verbose,
        )
    if size_usd < min_sz:
        addlog(
            f"[SKIP][{window_name} {cfg['window_size']}] size=${size_usd:.2f} < min=${min_sz:.2f}",
            verbose_int=2,
            verbose_state=verbose,
        )
        return False

    sz_pct = (size_usd / capital * 100) if capital else 0.0

    addlog(
        f"[BUY][{window_name} {cfg['window_size']}] p={meta['p_buy']:.3f}, base={base*100:.2f}% → size={sz_pct:.2f}% (cap=${size_usd:.2f})",
        verbose_int=1,
        verbose_state=verbose,
    )

    result = {
        "size_usd": size_usd,
        "window_name": window_name,
        "window_size": cfg["window_size"],
        **meta,
    }
    if "timestamp" in series.columns:
        result["created_ts"] = int(candle["timestamp"])
    result["created_idx"] = t
    return result
