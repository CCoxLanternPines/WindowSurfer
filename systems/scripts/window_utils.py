from __future__ import annotations

"""Helpers for window-based price normalisation."""

from typing import Tuple, Dict, Any
import pandas as pd

from systems.utils.addlog import addlog


def _parse_window_size(window_size: str) -> int:
    """Return number of candles corresponding to ``window_size``.

    Supported suffixes:
    ``h`` for hours, ``d`` for days, ``w`` for weeks. The dataset is assumed to
    use one-hour candles, so values are converted accordingly.
    """
    if not window_size:
        return 0
    try:
        value = int(window_size[:-1])
    except ValueError:  # pragma: no cover - invalid config
        value = 0
    unit = window_size[-1].lower()
    if unit == "h":
        factor = 1
    elif unit == "d":
        factor = 24
    elif unit == "w":
        factor = 24 * 7
    else:  # pragma: no cover - unsupported unit
        factor = 0
    return value * factor


def get_window_bounds(series: pd.DataFrame, t: int, window_size: str) -> Tuple[float, float]:
    """Return the ``(low, high)`` price bounds for a trailing window.

    The window spans ``window_size`` ending at index ``t`` (inclusive). If the
    window extends before the start of ``series`` the available range is used.
    """
    span = _parse_window_size(window_size)
    start = max(0, t - span + 1)
    window = series.iloc[start : t + 1]
    win_low = float(window["low"].min()) if "low" in window else float(window["close"].min())
    win_high = float(window["high"].max()) if "high" in window else float(window["close"].max())
    return win_low, win_high


def get_window_position(price: float, win_low: float, win_high: float) -> float:
    """Normalise ``price`` within the window bounds to a 0..1 range."""
    if win_high == win_low:
        return 0.5
    return max(0.0, min(1.0, (price - win_low) / (win_high - win_low)))


def check_buy_conditions(
    candle: Any,
    window_data: Dict[str, float],
    settings: Dict[str, Any],
    ledger_state: Dict[str, Any],
) -> tuple[bool, Dict[str, float]]:
    """Return whether buy conditions are met and related metadata."""

    price = float(candle["close"])
    win_low = float(window_data.get("low", 0.0))
    win_high = float(window_data.get("high", 0.0))
    p = get_window_position(price, win_low, win_high)

    verbose = ledger_state.get("verbose", 0)
    window_name = settings.get("window_name", "")
    unlock_map = ledger_state.setdefault("buy_unlock_p", {})
    open_notes = ledger_state.get("open_notes", [])

    unlock_p = unlock_map.get(window_name)
    if unlock_p is not None and open_notes:
        if p >= unlock_p:
            addlog(
                f"[UNLOCK][{window_name} {settings.get('window_size', '')}] p={p:.3f} >= unlock_p={unlock_p:.3f} → buys re-enabled",
                verbose_int=2,
                verbose_state=verbose,
            )
            unlock_map.pop(window_name, None)
        else:
            addlog(
                f"[GATE][{window_name} {settings.get('window_size', '')}] buy blocked; p={p:.3f} < unlock_p={unlock_p:.3f}",
                verbose_int=2,
                verbose_state=verbose,
            )
            return False, {}

    trigger = settings.get("buy_trigger_position", 0.0)
    if p > trigger:
        addlog(
            f"[SKIP][{window_name} {settings.get('window_size', '')}] p={p:.3f} > buy_trigger={trigger:.3f}",
            verbose_int=3,
            verbose_state=verbose,
        )
        return False, {}

    p_target = settings.get("maturity_position", 1.0)
    price_target = win_low + p_target * (win_high - win_low)
    roi_target = (price_target - price) / price if price else 0.0
    unlock_new = min(1.0, p + settings.get("reset_buy_percent", 0.0))

    return True, {
        "p_buy": p,
        "target_price": price_target,
        "target_roi": roi_target,
        "unlock_p": unlock_new,
    }


def check_sell_conditions(
    candle: Any,
    note: Dict[str, Any],
    settings: Dict[str, Any],
    ledger_state: Dict[str, Any],
) -> bool:
    """Return ``True`` if the note should be sold on this candle."""

    price = float(candle["close"])
    verbose = ledger_state.get("verbose", 0)
    window_name = ledger_state.get("window_name", "")
    window_size = ledger_state.get("window_size", "")

    if ledger_state.get("sell_count", 0) >= ledger_state.get(
        "max_sells", settings.get("max_notes_sell_per_candle", 1)
    ):
        return False

    if price < note.get("target_price", float("inf")) or price < note.get("entry_price", float("inf")):
        return False

    buy = note.get("entry_price", 0.0)
    qty = note.get("entry_amount", 0.0)
    target = note.get("target_price", 0.0)
    roi = (price - buy) / buy if buy else 0.0

    maturity_pos = settings.get("maturity_position", 1.0)
    if maturity_pos >= 0.99 and target < buy:
        addlog(
            f"[WARN] SellTargetBelowBuy note=#{note.get('id', '')} buy=${buy:.4f} target=${target:.4f}",
            verbose_int=1,
            verbose_state=verbose,
        )
    if roi > 5.0:
        addlog(
            f"[WARN] UnusuallyHighROI note=#{note.get('id', '')} roi={roi*100:.2f}% (check scale/decimals)",
            verbose_int=1,
            verbose_state=verbose,
        )

    addlog(
        f"[SELL][{window_name} {window_size}] note=#{note.get('id', '')} qty={qty:.6f} buy=${buy:.4f} now=${price:.4f} target=${target:.4f} roi={roi*100:.2f}%",
        verbose_int=1,
        verbose_state=verbose,
    )

    ledger_state["sell_count"] = ledger_state.get("sell_count", 0) + 1
    return True


def get_trade_params(
    current_price,
    window_high,
    window_low,
    config,
    entry_price=None,
):
    """Compute position metrics and multipliers for buy/sell decisions.

    Parameters
    ----------
    current_price:
        Current market price.
    window_high:
        Upper boundary of the price window.
    window_low:
        Lower boundary of the price window.
    config:
        Strategy configuration containing multiplier scales and optional
        ``dead_zone_pct`` and ``maturity_multiplier`` values.
    entry_price:
        Optional entry price used to calculate maturity ROI.

    Returns
    -------
    dict
        Dictionary containing ``pos_pct``, ``in_dead_zone``, buy multipliers,
        buy/sell cooldown multipliers, and ``maturity_roi`` (``None`` if
        ``entry_price`` is not provided).
    """

    window_range = window_high - window_low
    if window_range == 0:
        pos_pct = 0.0
    else:
        pos_pct = ((current_price - window_low) / window_range) * 2 - 1

    dead_zone_pct = config.get("dead_zone_pct", 0.0)
    dead_zone_half = dead_zone_pct / 2
    in_dead_zone = abs(pos_pct) <= dead_zone_half if dead_zone_pct > 0 else False

    buy_scale = config.get("buy_multiplier_scale", 1.0)
    buy_multiplier = 1.0 + (abs(pos_pct) * (buy_scale - 1.0))

    buy_cd_multiplier = 1.0 + (
        abs(pos_pct)
        * (config.get("buy_cooldown_multiplier_scale", 1.0) - 1.0)
    )
    sell_cd_multiplier = 1.0 + (
        abs(pos_pct)
        * (config.get("sell_cooldown_multiplier_scale", 1.0) - 1.0)
    )

    maturity_roi = None
    if entry_price is not None and window_range != 0:
        maturity_multiplier = config.get("maturity_multiplier", 1.0)
        mirrored_pos = -pos_pct
        target_price = window_low + ((mirrored_pos + 1) / 2) * window_range
        maturity_roi = ((target_price - entry_price) / entry_price) * maturity_multiplier

    return {
        "pos_pct": pos_pct,
        "in_dead_zone": in_dead_zone,
        "buy_multiplier": buy_multiplier,
        "buy_cooldown_multiplier": buy_cd_multiplier,
        "sell_cooldown_multiplier": sell_cd_multiplier,
        "maturity_roi": maturity_roi,
    }
