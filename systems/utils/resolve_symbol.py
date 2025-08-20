"""Helpers for resolving exchange symbols and tag conversions."""

from __future__ import annotations

import ccxt


def resolve_symbols(client: ccxt.Exchange, config_name: str) -> dict[str, str]:
    """Validate ``config_name`` and return normalized symbols.

    Parameters
    ----------
    client:
        Initialized CCXT Kraken client.
    config_name:
        Human friendly market string such as ``"DOGE/USD"``.
    """
    markets = client.load_markets()

    if config_name not in markets:
        raise RuntimeError(f"[ERROR] Invalid trading pair: {config_name}")

    m = markets[config_name]
    kraken_name = m["symbol"]
    kraken_pair = m["id"]

    base, quote = m["base"], m["quote"]
    if quote == "USD":
        binance_quote = "USDT"
    else:
        binance_quote = quote
    binance_name = base + binance_quote

    return {
        "kraken_name": kraken_name,
        "kraken_pair": kraken_pair,
        "binance_name": binance_name,
    }


# ---------------------------------------------------------------------------
# Legacy helpers retained for compatibility

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
    """Return base symbol and Kraken quote asset code for ``tag``."""
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
