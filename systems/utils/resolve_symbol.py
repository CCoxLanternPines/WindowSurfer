"""Helpers for resolving symbol and ledger configuration."""

from systems.utils.config import load_settings
from systems.utils.quote_norm import norm_quote


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
    ledger = resolve_ledger_settings(tag)
    return {
        "kraken": ledger["kraken_name"],
        "binance": ledger["binance_name"],
    }


def resolve_ccxt_symbols(settings: dict, ledger: str) -> tuple[str, str]:
    """Return Kraken and Binance symbols for ``ledger`` from ``settings``."""
    ledgers = settings.get("ledger_settings", {})
    if ledger not in ledgers:
        raise ValueError(f"Ledger '{ledger}' not found in settings")
    cfg = ledgers[ledger]
    kraken = cfg.get("kraken_name", "")
    if "/" in kraken:
        base, quote = kraken.split("/", 1)
        kraken = f"{base}/{norm_quote(quote)}"
    else:
        kraken = norm_quote(kraken)

    binance = cfg.get("binance_name", "")
    if "/" in binance:
        base_b, quote_b = binance.split("/", 1)
        binance = f"{base_b}/{norm_quote(quote_b)}"

    return kraken, binance


def to_tag(symbol: str) -> str:
    """Normalize exchange symbol to uppercase tag without separators."""
    return symbol.replace("/", "").replace(" ", "").upper()


def sim_path_csv(tag: str) -> str:
    """Return canonical SIM CSV path for ``tag``."""
    return f"data/sim/{tag}_1h.csv"


def live_path_csv(tag: str) -> str:
    """Return canonical LIVE CSV path for ``tag``."""
    return f"data/live/{tag}_1h.csv"


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
