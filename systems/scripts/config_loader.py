"""Configuration loading utilities.

Combines global settings with ledger-specific settings and enriches the
result with dynamically resolved exchange information from the wallet
cache.
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover - fallback if PyYAML is unavailable
    yaml = None

from .wallet_cache import resolve_pairs, get_exchange_precision, load_wallet_cache


def load_global_settings() -> Dict[str, Any]:
    """Load ``config/global_settings.yaml`` into a dictionary."""
    path = os.path.join("config", "global_settings.yaml")
    with open(path, "r", encoding="utf-8") as fh:
        content = fh.read()
    if yaml is not None:
        return yaml.safe_load(content) or {}
    return json.loads(content or "{}")


def load_ledger_settings(ledger_name: str) -> Dict[str, Any]:
    """Load ``ledgers/{ledger_name}.json`` into a dictionary."""
    path = os.path.join("ledgers", f"{ledger_name}.json")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def merge_settings(global_cfg: Dict[str, Any], ledger_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Merge global and ledger settings.

    Ledger values override global ones. The resulting configuration includes
    the ledger name and coin list.
    """
    merged = _deep_merge(global_cfg, ledger_cfg)
    merged["ledger_name"] = ledger_cfg.get("name")
    if "coins" not in merged:
        merged["coins"] = {}
    return merged


def load_runtime_config(ledger_name: str) -> Dict[str, Any]:
    """Load the merged runtime configuration for ``ledger_name``.

    The configuration is enriched with exchange pair information and order
    precision details for each coin defined in the ledger.
    """
    global_cfg = load_global_settings()
    ledger_cfg = load_ledger_settings(ledger_name)
    cfg = merge_settings(global_cfg, ledger_cfg)

    # Ensure wallet cache is loaded so missing files raise early.
    load_wallet_cache()

    fiat = cfg.get("fiat", "USD")
    for coin, coin_cfg in cfg.get("coins", {}).items():
        pairs = resolve_pairs(coin, fiat)
        binance_info = get_exchange_precision(coin, fiat, "binance")
        kraken_info = get_exchange_precision(coin, fiat, "kraken")

        coin_cfg["binance"] = {"pair": pairs["binance"], **binance_info}
        coin_cfg["kraken"] = {"pair": pairs["kraken"], **kraken_info}

    return cfg

