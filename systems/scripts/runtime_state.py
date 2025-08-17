from __future__ import annotations

from typing import Any, Dict, Optional

from systems.scripts.execution_handler import load_or_fetch_snapshot
from systems.utils.resolve_symbol import split_tag


def build_runtime_state(
    settings: Dict[str, Any],
    ledger_cfg: Dict[str, Any],
    mode: str,
    *,
    prev: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build and refresh runtime state for engines.

    Parameters
    ----------
    settings:
        Global settings dictionary.
    ledger_cfg:
        Ledger configuration mapping containing at least ``tag``.
    mode:
        Either ``"sim"`` or ``"live"``.
    prev:
        Previous runtime state to carry over verbose level and unlock map.
    """

    prev = prev or {}
    general = settings.get("general_settings", {})

    limits = prev.get(
        "limits",
        {
            "min_note_size": float(general.get("minimum_note_size", 0.0)),
            "max_note_usdt": float(general.get("max_note_usdt", float("inf"))),
        },
    )

    defaults = {
        "window_size": 24,
        "window_step": 2,
        "buy_trigger": 3.0,
        "sell_trigger": 10.0,
        "flat_sell_fraction": 0.2,
        "flat_sell_threshold": 0.5,
        "max_pressure": 10.0,
        "strong_move_threshold": 0.15,
        "range_min": 0.08,
        "volume_skew_bias": 0.4,
        "flat_band_deg": 10.0,
    }
    strategy_cfg = {**defaults, **general.get("strategy_settings", {})}

    buy_unlock_p = prev.get("buy_unlock_p", {})
    verbose = prev.get("verbose", 0)

    if mode == "sim":
        capital = float(settings.get("simulation_capital", 0.0))
    elif mode == "live":
        tag = ledger_cfg.get("tag", "")
        snapshot = load_or_fetch_snapshot(tag)
        _, quote = split_tag(tag)
        balance = snapshot.get("balance", {})
        capital = float(balance.get(quote, 0.0))
    else:
        capital = prev.get("capital", 0.0)

    state = {
        "capital": capital,
        "buy_unlock_p": buy_unlock_p,
        "verbose": verbose,
        "limits": limits,
        "strategy": strategy_cfg,
    }

    state.setdefault("pressures", prev.get("pressures", {"buy": {}, "sell": {}}))
    state.setdefault("last_features", prev.get("last_features", {}))

    return state
