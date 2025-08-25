from __future__ import annotations

"""Unified configuration loader for accounts and strategy settings."""

from typing import Any, Dict

from systems.utils.config import (
    load_general,
    load_account_settings,
    load_coin_settings,
    load_keys,
    resolve_coin_config,
    resolve_account_market,
)

_CONFIG_CACHE: Dict[str, Any] | None = None


def load_config(*, reload: bool = False) -> Dict[str, Any]:
    """Load accounts and strategy settings from JSON files.

    Returns
    -------
    dict
        Dictionary with ``accounts`` and ``general_settings`` entries.
    """
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None or reload:
        general = load_general(reload=reload)
        coin_settings = load_coin_settings(reload=reload)
        accounts_cfg = load_account_settings(reload=reload)
        keys = load_keys(reload=reload)

        accounts: Dict[str, Any] = {}
        for acct_name, acct_cfg in accounts_cfg.items():
            merged_markets: Dict[str, Any] = {}
            for market in acct_cfg.get("market settings", {}).keys():
                coin_cfg = resolve_coin_config(market, coin_settings)
                acct_mkt_cfg = resolve_account_market(acct_name, market, accounts_cfg)
                merged_markets[market] = {**coin_cfg, **acct_mkt_cfg}
            accounts[acct_name] = {
                "api_key": keys.get(acct_name, {}).get("api_key", ""),
                "api_secret": keys.get(acct_name, {}).get("api_secret", ""),
                "is_live": acct_cfg.get("is_live", False),
                "reporting": acct_cfg.get("reporting", {}),
                "markets": merged_markets,
            }
        _CONFIG_CACHE = {"accounts": accounts, "general_settings": general}
    return _CONFIG_CACHE
