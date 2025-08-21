from __future__ import annotations

"""Configuration helper utilities."""

from pathlib import Path
import json
from typing import Any, Dict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SETTINGS_CACHE: Dict[str, Any] | None = None
_ACCOUNT_CACHE: Dict[str, Any] | None = None
_COIN_CACHE: Dict[str, Any] | None = None


def resolve_path(rel_path: str) -> Path:
    """Return an absolute path for ``rel_path`` from the project root."""
    return _PROJECT_ROOT / rel_path


def load_settings(*, reload: bool = False) -> Dict[str, Any]:
    """Load global settings from ``settings/settings.json``."""
    global _SETTINGS_CACHE
    if _SETTINGS_CACHE is None or reload:
        settings_path = resolve_path("settings/settings.json")
        with settings_path.open("r", encoding="utf-8") as fh:
            _SETTINGS_CACHE = json.load(fh)
    return _SETTINGS_CACHE


def load_account_settings(*, reload: bool = False) -> Dict[str, Any]:
    """Return account configuration mapping from ``account_settings.json``."""
    global _ACCOUNT_CACHE
    if _ACCOUNT_CACHE is None or reload:
        acct_path = resolve_path("settings/account_settings.json")
        with acct_path.open("r", encoding="utf-8") as fh:
            _ACCOUNT_CACHE = json.load(fh)
    return _ACCOUNT_CACHE


def load_coin_settings(*, reload: bool = False) -> Dict[str, Any]:
    """Return coin configuration mapping from ``coin_settings.json``."""
    global _COIN_CACHE
    if _COIN_CACHE is None or reload:
        coin_path = resolve_path("settings/coin_settings.json")
        with coin_path.open("r", encoding="utf-8") as fh:
            _COIN_CACHE = json.load(fh)
    return _COIN_CACHE
