"""Helpers for resolving symbol and ledger configuration."""

from systems.utils.settings_loader import load_settings
from systems.utils.symbol_mapper import get_symbol_config

SETTINGS = load_settings()


def resolve_ledger_settings(tag: str, settings: dict | None = None) -> dict:
    """Return ledger configuration matching ``tag``."""
    cfg = settings or SETTINGS
    tag = tag.upper()
    for ledger in cfg.get("ledger_settings", {}).values():
        if ledger.get("tag") == tag:
            return ledger
    raise ValueError(f"No ledger found for tag: {tag}")


def resolve_symbol(tag: str) -> dict:
    """Resolve exchange-specific pair names for ``tag``."""
    cfg = get_symbol_config(tag)
    return {
        "kraken": cfg["kraken"]["wsname"],
        "binance": cfg["binance"]["symbol"],
    }
