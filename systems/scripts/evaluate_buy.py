from __future__ import annotations

"""Position-based buy evaluation."""

from typing import Dict, Any

from systems.scripts.window_utils import get_window_bounds, get_window_position
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

    win_low, win_high = get_window_bounds(series, t, cfg["window_size"])
    price = float(series.iloc[t]["close"])
    p = get_window_position(price, win_low, win_high)

    unlock_map = runtime_state.setdefault("buy_unlock_p", {})
    unlock_p = unlock_map.get(window_name)
    if unlock_p is not None:
        if p >= unlock_p:
            addlog(
                f"[UNLOCK][{window_name} {cfg['window_size']}] p={p:.3f} >= unlock_p={unlock_p:.3f} → buys re-enabled",
                verbose_int=2,
                verbose_state=verbose,
            )
            unlock_map.pop(window_name, None)
        else:
            addlog(
                f"[GATE][{window_name} {cfg['window_size']}] buy blocked; p={p:.3f} < unlock_p={unlock_p:.3f}",
                verbose_int=2,
                verbose_state=verbose,
            )
            return False

    open_notes = []
    if ledger:
        open_notes = [
            n for n in ledger.get_open_notes() if n.get("window_name") == window_name
        ]
    if len(open_notes) >= cfg.get("max_open_notes", 0):
        return False

    trigger = cfg.get("buy_trigger_position", 0.0)
    if p > trigger:
        addlog(
            f"[SKIP][{window_name} {cfg['window_size']}] p={p:.3f} > buy_trigger={trigger:.3f}",
            verbose_int=3,
            verbose_state=verbose,
        )
        return False

    capital = runtime_state.get("capital", 0.0)
    base = cfg.get("investment_fraction", 0.0)
    mult = 1 + (1 - p) * (cfg.get("window_transform_multiplier", 1.0) - 1)
    size_usd = capital * base * mult

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
        f"[BUY][{window_name} {cfg['window_size']}] p={p:.3f}, base={base*100:.2f}%, mult={mult:.2f}x → size={sz_pct:.2f}% (cap=${size_usd:.2f})",
        verbose_int=1,
        verbose_state=verbose,
    )

    unlock_p = min(1.0, p + cfg.get("reset_buy_percent", 0.0))
    p_target = cfg.get("maturity_position", 1.0)
    price_target = win_low + p_target * (win_high - win_low)
    roi_target = (price_target - price) / price if price else 0.0

    result = {
        "size_usd": size_usd,
        "window_name": window_name,
        "window_size": cfg["window_size"],
        "p_buy": p,
        "target_price": price_target,
        "target_roi": roi_target,
        "unlock_p": unlock_p,
    }
    if "timestamp" in series.columns:
        result["created_ts"] = int(series.iloc[t]["timestamp"])
    result["created_idx"] = t
    return result
