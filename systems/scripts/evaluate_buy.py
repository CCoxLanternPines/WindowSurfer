from __future__ import annotations

"""Position-based buy evaluation."""

from typing import Dict, Any

from systems.scripts.window_utils import get_window_bounds, get_window_position
from systems.utils.addlog import addlog


def evaluate_buy_signal(
    ctx: Dict[str, Any],
    t: int,
    series,
    cfg: Dict[str, Any],
    runtime_state: Dict[str, Any],
):
    """Return buy sizing and note metadata when conditions are met.

    Parameters
    ----------
    ctx:
        Context dictionary containing at least a ``ledger`` instance and, if
        multiple strategies are used, a ``window`` name.
    t:
        Current candle index within ``series``.
    series:
        Candle DataFrame with at least ``close``, ``low`` and ``high`` columns.
    cfg:
        Strategy configuration.
    runtime_state:
        Mutable dictionary carrying ``capital`` and ``buy_unlock_p``.
    """

    ledger = ctx.get("ledger")
    verbose = runtime_state.get("verbose", 0)

    win_low, win_high = get_window_bounds(series, t, cfg["window_size"])
    price = float(series.iloc[t]["close"])
    p = get_window_position(price, win_low, win_high)

    unlock_p = runtime_state.get("buy_unlock_p")
    if unlock_p is not None:
        if p >= unlock_p:
            addlog(
                f"[UNLOCK] p={p:.3f} >= unlock_p={unlock_p:.3f} → buys re-enabled",
                verbose_int=2,
                verbose_state=verbose,
            )
            runtime_state["buy_unlock_p"] = None
        else:
            addlog(
                f"[GATE] buy blocked; p={p:.3f} < unlock_p={unlock_p:.3f}",
                verbose_int=2,
                verbose_state=verbose,
            )
            return False

    open_notes = ledger.get_open_notes() if ledger else []
    if len(open_notes) >= cfg.get("max_open_notes", 0):
        return False

    trigger = cfg.get("buy_trigger_position", 0.0)
    if p > trigger:
        addlog(
            f"[SKIP] p={p:.3f} > buy_trigger={trigger:.3f}",
            verbose_int=3,
            verbose_state=verbose,
        )
        return False

    capital = runtime_state.get("capital", 0.0)
    base = cfg.get("investment_fraction", 0.0)
    mult = 1 + (1 - p) * (cfg.get("window_transform_multiplier", 1.0) - 1)
    size_usd = capital * base * mult

    addlog(
        f"[BUY] p={p:.3f}, base={base*100:.2f}%, mult={mult:.2f}x → size={base*mult*100:.2f}% (cap=${size_usd:.2f})",
        verbose_int=1,
        verbose_state=verbose,
    )

    unlock_p = min(1.0, p + cfg.get("reset_buy_percent", 0.0))
    p_target = cfg.get("maturity_position", 1.0)
    price_target = win_low + p_target * (win_high - win_low)
    roi_target = (price_target - price) / price if price else 0.0

    note_meta = {
        "p_buy": p,
        "unlock_p": unlock_p,
        "target_price": price_target,
        "target_roi": roi_target,
        "created_idx": t,
    }
    if "timestamp" in series.columns:
        note_meta["created_ts"] = int(series.iloc[t]["timestamp"])

    return {"size_usd": size_usd, "note": note_meta}
