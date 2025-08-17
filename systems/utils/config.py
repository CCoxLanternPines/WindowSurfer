from __future__ import annotations

"""Configuration helper utilities."""

from pathlib import Path
import json
from typing import Any, Dict

from systems.utils.addlog import addlog

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SETTINGS_CACHE: Dict[str, Any] | None = None


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


def resolve_ccxt_symbols_by_coin(coin: str) -> tuple[str, str]:
    """Return Kraken and Binance symbols for ``coin``.

    The first ledger whose tag starts with ``coin`` (case-insensitive) is used.
    Logs which ledger tag was matched.
    """

    settings = load_settings()
    coin_up = coin.upper()
    for cfg in settings.get("ledger_settings", {}).values():
        tag = cfg.get("tag", "")
        if tag.upper().startswith(coin_up):
            kraken = cfg.get("kraken_name", "")
            binance = cfg.get("binance_name", "")
            addlog(
                f"[CONFIG] coin={coin_up} resolved using tag={tag}",
                verbose_int=1,
                verbose_state=True,
            )
            return kraken, binance
    msg = (
        f"[ERROR] No ledger maps coin={coin_up} to exchange symbols. "
        f"Add a ledger with tag starting '{coin_up}' including kraken_name/binance_name."
    )
    addlog(msg, verbose_int=1, verbose_state=True)
    raise ValueError(msg)
