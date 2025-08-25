from __future__ import annotations

"""Coin settings loader merging defaults with market overrides."""

import json
from pathlib import Path
from typing import Any, Dict

COIN_SETTINGS_PATH = Path(__file__).resolve().parents[2] / "settings" / "coin_settings.json"


def load_coin_settings(market: str) -> Dict[str, Any]:
    """
    Load coin-specific settings from coin_settings.json.
    Merge 'default' with overrides for given market.
    """
    with COIN_SETTINGS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f).get("coin_settings", {})

    cfg = dict(data.get("default", {}))
    overrides = data.get(market)
    if overrides:
        cfg.update(overrides)
    return cfg


def get_coin_setting(coin: str, key: str, default: Any = None) -> Any:
    """Retrieve a single setting for ``coin`` with optional ``default``."""
    cfg = load_coin_settings(coin)
    return cfg.get(key, default)
