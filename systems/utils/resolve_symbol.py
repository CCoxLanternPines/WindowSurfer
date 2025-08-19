"""Helpers for resolving exchange symbols and tag conversions."""

from __future__ import annotations

BASE_MAP = {
    "XBT": "BTC",
    "XDG": "DOGE",
}


def resolve_symbols(kraken_name: str) -> dict[str, str]:
    """Derive related symbol names from a Kraken CCXT pair."""

    base, quote = kraken_name.split("/")
    base_cc = BASE_MAP.get(base, base)

    kraken_pair = base + quote

    if quote == "USD":
        binance = base_cc + "USDT"
    else:
        binance = base_cc + quote

    return {
        "kraken_name": f"{base}/{quote}",
        "kraken_pair": kraken_pair,
        "binance_name": binance,
    }


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

