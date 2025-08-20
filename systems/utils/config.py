from __future__ import annotations

"""Basic configuration utilities and path helpers."""

from pathlib import Path
import json
from typing import Any, Dict

import ccxt

from systems.utils.addlog import addlog
from systems.utils.resolve_symbol import resolve_symbols, to_tag

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SETTINGS_CACHE: Dict[str, Any] | None = None


def resolve_path(rel_path: str) -> Path:
    """Return an absolute path for ``rel_path`` from the project root."""
    return _PROJECT_ROOT / rel_path


def load_settings(*, reload: bool = False) -> Dict[str, Any]:
    """Load ``settings/settings.json`` with optional caching."""
    global _SETTINGS_CACHE
    if _SETTINGS_CACHE is None or reload:
        settings_path = resolve_path("settings/settings.json")
        with settings_path.open("r", encoding="utf-8") as fh:
            _SETTINGS_CACHE = json.load(fh)
    return _SETTINGS_CACHE


def load_ledger_config(ledger_name: str) -> Dict[str, Any]:
    """Deprecated ledger loader retained for backward compatibility."""
    raise ValueError("load_ledger_config is deprecated; use load_config")


def resolve_ccxt_symbols_by_coin(coin: str) -> tuple[str, str]:
    """Return Kraken and Binance symbols for ``coin`` based on configured markets."""
    from systems.utils.load_config import load_config

    cfg = load_config()
    client = ccxt.kraken({"enableRateLimit": True})
    coin_up = coin.upper()
    for acct in cfg.get("accounts", {}).values():
        for market in acct.get("markets", {}).keys():
            symbols = resolve_symbols(client, market)
            tag = to_tag(symbols["kraken_name"])
            base = symbols["kraken_name"].split("/")[0].upper()
            if base.startswith(coin_up):
                addlog(
                    f"[CONFIG] coin={coin_up} resolved using tag={tag}",
                    verbose_int=1,
                    verbose_state=True,
                )
                return symbols["kraken_name"], symbols["binance_name"]
    msg = f"[ERROR] No market maps coin={coin_up} to exchange symbols"
    addlog(msg, verbose_int=1, verbose_state=True)
    raise ValueError(msg)
