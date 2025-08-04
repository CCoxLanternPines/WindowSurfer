"""Helpers for resolving symbol and ledger configuration."""

from systems.utils.settings_loader import load_settings


SETTINGS = load_settings()


def resolve_ledger_settings(ledger: str, settings: dict | None = None) -> dict:
    """Return ledger configuration for ``ledger`` key."""
    cfg = settings or SETTINGS
    return cfg.get("ledger_settings", {}).get(ledger, {})


def resolve_symbol(ledger: str) -> dict:
    """Resolve exchange-specific pair names for ``ledger``."""
    ledger_cfg = resolve_ledger_settings(ledger)
    return {
        "kraken": ledger_cfg.get("tag"),
        "binance": ledger_cfg.get("binance_name"),
    }


def split_tag(tag: str) -> tuple[str, str]:
    """Return base symbol and Kraken quote asset code for ``tag``.

    Parameters
    ----------
    tag:
        Trading pair tag such as ``DOGEUSD`` or ``SOLUSDC``.

    Returns
    -------
    tuple[str, str]
        A tuple ``(base, quote_asset)`` where ``base`` is the base currency
        symbol and ``quote_asset`` is the Kraken asset code for the quote
        currency (e.g. ``"ZUSD"`` for USD).
    """
    tag = tag.upper()
    mapping = {
        "USDT": "USDT",
        "USDC": "USDC",
        "DAI": "DAI",
        "USD": "ZUSD",
        "EUR": "ZEUR",
        "GBP": "ZGBP",
    }
    for suffix, asset_code in mapping.items():
        if tag.endswith(suffix):
            return tag[: -len(suffix)], asset_code
    return tag, ""
