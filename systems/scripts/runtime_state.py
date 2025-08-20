from __future__ import annotations

from typing import Any, Dict, Optional

from systems.utils.resolve_symbol import resolve_symbols
from systems.utils.config import (
    resolve_coin_config,
    resolve_account_market,
)


def build_runtime_state(
    general: Dict[str, Any],
    coin_settings: Dict[str, Any],
    accounts_cfg: Dict[str, Any],
    account: str,
    symbol: str,
    mode: str,
    *,
    client,
    prev: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build and refresh runtime state for engines."""

    prev = prev or {}
    coin_cfg = resolve_coin_config(symbol, coin_settings)
    acct_mkt_cfg = resolve_account_market(account, symbol, accounts_cfg)
    knobs = {**coin_cfg, **acct_mkt_cfg}

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
    strategy = {**defaults, **knobs}

    buy_unlock_p = prev.get("buy_unlock_p", {})
    verbose = prev.get("verbose", 0)

    symbols = resolve_symbols(client, symbol)
    if mode == "sim":
        capital = float(general.get("simulation_capital", 0.0))
    elif mode == "live":
        quote = symbols["kraken_name"].split("/")[1]
        balance = client.fetch_balance().get("free", {})
        capital = float(balance.get(quote, 0.0))
    else:
        capital = prev.get("capital", 0.0)

    state = {
        "capital": capital,
        "buy_unlock_p": buy_unlock_p,
        "verbose": verbose,
        "limits": limits,
        "strategy": strategy,
    }

    state.update(symbols)

    state.setdefault("pressures", prev.get("pressures", {"buy": {}, "sell": {}}))
    state.setdefault("last_features", prev.get("last_features", {}))

    return state
