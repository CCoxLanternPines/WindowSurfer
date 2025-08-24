"""Helpers for resolving exchange symbols and tag conversions."""

from __future__ import annotations

import ccxt


def resolve_symbols(client: ccxt.Exchange, config_name: str) -> dict[str, str]:
    """Return exchange identifiers for ``config_name`` using CCXT."""

    markets = client.load_markets()

    if config_name not in markets:
        raise RuntimeError(f"[ERROR] Invalid trading pair: {config_name}")

    m = markets[config_name]

    return {
        "kraken_name": m["symbol"],
        "kraken_pair": m["id"],
        "binance_name": (
            f"{m['base']}USDT" if m["quote"] == "USD" else f"{m['base']}{m['quote']}"
        ),
    }


def candle_filename(account: str, market: str, live: bool = False) -> str:
    """Return canonical candle CSV path for ``account``/``market``."""

    base = f"{account}_{market.replace('/', '_')}_1h.csv"
    folder = "data/live" if live else "data/sim"
    return f"{folder}/{base}"


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
