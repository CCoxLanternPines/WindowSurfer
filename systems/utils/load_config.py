from __future__ import annotations

"""Unified configuration loader for accounts and strategy settings."""

import json
from typing import Any, Dict

from systems.utils.config import resolve_path

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
        settings_path = resolve_path("settings/settings.json")
        accounts_path = resolve_path("settings/accounts.json")
        with settings_path.open("r", encoding="utf-8") as fh:
            settings = json.load(fh)
        with accounts_path.open("r", encoding="utf-8") as fh:
            accounts_raw = json.load(fh)
        general = settings.get("general_settings", {})
        coin_settings = settings.get("coin_settings", {})
        default = coin_settings.get("default", {})
        accounts: Dict[str, Any] = {}
        for acct_name, acct_cfg in accounts_raw.items():
            merged_markets: Dict[str, Any] = {}
            for market in acct_cfg.get("markets", []):
                strat = dict(default)
                strat.update(coin_settings.get(market, {}))
                merged_markets[market] = strat
            accounts[acct_name] = {
                "api_key": acct_cfg.get("api_key", ""),
                "api_secret": acct_cfg.get("api_secret", ""),
                "markets": merged_markets,
            }
        _CONFIG_CACHE = {"accounts": accounts, "general_settings": general}
    return _CONFIG_CACHE
