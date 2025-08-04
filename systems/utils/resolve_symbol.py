"""Helpers for resolving symbol and ledger configuration."""

from systems.utils.settings_loader import load_settings


SETTINGS = load_settings()


def resolve_ledger_settings(ledger_name: str, settings: dict | None = None) -> dict:
    """Return ledger configuration for ``ledger_name``."""
    cfg = settings or SETTINGS
    try:
        return cfg.get("ledger_settings", {})[ledger_name]
    except KeyError:
        raise ValueError(f"No ledger found for name: {ledger_name}") from None


def resolve_symbol(ledger_name: str) -> dict:
    """Resolve exchange-specific pair names for ``ledger_name``."""
    ledger = resolve_ledger_settings(ledger_name)
    return {
        "kraken": ledger["kraken_name"],
        "binance": ledger["binance_name"],
    }
