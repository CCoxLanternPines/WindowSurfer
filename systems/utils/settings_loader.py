from __future__ import annotations

"""Coin settings loader merging defaults with market overrides."""

import json
from pathlib import Path
from typing import Any, Dict

COIN_SETTINGS_PATH = Path(__file__).resolve().parents[2] / "settings" / "coin_settings.json"
GENERAL_SETTINGS_PATH = Path(__file__).resolve().parents[2] / "settings" / "settings.json"
ACCOUNT_SETTINGS_PATH = Path(__file__).resolve().parents[2] / "settings" / "account_settings.json"


def load_coin_settings(market: str) -> Dict[str, Any]:
    """Load coin-specific settings merging defaults with market overrides."""
    with COIN_SETTINGS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f).get("coin_settings", {})

    cfg = dict(data.get("default", {}))
    overrides = data.get(market)
    if overrides:
        cfg.update(overrides)
    return cfg


def load_general_settings() -> Dict[str, Any]:
    """Return general simulation settings."""
    with GENERAL_SETTINGS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f).get("general_settings", {})


def load_account_settings() -> Dict[str, Any]:
    """Return account configuration mapping."""
    with ACCOUNT_SETTINGS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f).get("accounts", {})
