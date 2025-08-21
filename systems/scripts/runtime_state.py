from __future__ import annotations

from typing import Any, Dict, Optional

from systems.utils.config import (
    load_settings,
    load_account_settings,
    load_coin_settings,
)


def build_state(
    account_name: str,
    market: str,
    mode: str,
    *,
    prev: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build and refresh runtime state for engines."""

    prev = prev or {}
    settings = load_settings()
    accounts = load_account_settings()
    coins = load_coin_settings()

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
        "flat_sell_percent": 0.25,
        "all_sell_count": 99,
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
    else:
        account_cfg = accounts.get(account_name, {})
        capital = float(account_cfg.get("capital", 0.0))

    coin_cfg = coins.get(market, {})

    state = {
        "capital": capital,
        "buy_unlock_p": buy_unlock_p,
        "verbose": verbose,
        "limits": limits,
        "strategy": strategy_cfg,
        "symbol": coin_cfg.get("tag", ""),
    }

    state.setdefault("pressures", prev.get("pressures", {"buy": {}, "sell": {}}))
    state.setdefault("last_features", prev.get("last_features", {}))

    return state
