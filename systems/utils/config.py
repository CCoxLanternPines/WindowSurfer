from __future__ import annotations

"""Configuration helper utilities."""

from pathlib import Path
import json
from typing import Any, Dict

from systems.utils.addlog import addlog

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SETTINGS_CACHE: Dict[str, Any] | None = None
_DEPRECATION_WARNED = False

_DEPRECATED_KEYS = {
    "buy_cooldown",
    "sell_cooldown",
    "maturity_multiplier",
    "buy_multiplier_scale",
    "buy_cooldown_multiplier_scale",
    "sell_cooldown_multiplier_scale",
    "dead_zone_pct",
    "buy_floor",
    "sell_ceiling",
    "cooldown",
}


def _warn_deprecated(settings: Dict[str, Any]) -> None:
    global _DEPRECATION_WARNED
    if _DEPRECATION_WARNED:
        return
    found = set()
    for ledger in settings.get("ledger_settings", {}).values():
        for win in ledger.get("window_settings", {}).values():
            found.update(key for key in win if key in _DEPRECATED_KEYS)
    if found:
        addlog(
            f"[WARN] Deprecated config keys detected: {', '.join(sorted(found))}",
            verbose_int=1,
            verbose_state=True,
        )
        _DEPRECATION_WARNED = True


def resolve_path(rel_path: str) -> Path:
    """Return an absolute path for ``rel_path`` from the project root."""
    return _PROJECT_ROOT / rel_path


def load_settings(*, reload: bool = False) -> Dict[str, Any]:
    """Load settings from ``settings/settings.json`` with optional caching."""
    global _SETTINGS_CACHE
    if _SETTINGS_CACHE is None or reload:
        settings_path = resolve_path("settings/settings.json")
        with settings_path.open("r", encoding="utf-8") as fh:
            _SETTINGS_CACHE = json.load(fh)
        _warn_deprecated(_SETTINGS_CACHE)
    return _SETTINGS_CACHE


def load_ledger_config(ledger_name: str) -> Dict[str, Any]:
    """Return the configuration dictionary for a given ledger.

    Parameters
    ----------
    ledger_name: str
        Name of the ledger to load from settings.

    Raises
    ------
    ValueError
        If the requested ledger does not exist in ``settings.json``.
    """

    settings = load_settings()
    ledgers = settings.get("ledger_settings", {})
    if ledger_name not in ledgers:
        raise ValueError(f"Ledger '{ledger_name}' not found in settings")
    return ledgers[ledger_name]
