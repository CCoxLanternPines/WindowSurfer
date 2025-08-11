"""Helpers for resolving symbol and ledger configuration."""

from pathlib import Path
import json

import ccxt

from systems.utils.config import load_settings
from systems.utils.addlog import addlog


SETTINGS = load_settings()


CACHE_DIR = Path("data/cache")
KRAKEN_FILE = CACHE_DIR / "kraken_markets.json"
BINANCE_FILE = CACHE_DIR / "binance_markets.json"


def refresh_pair_cache(verbose: int = 0) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    kraken = ccxt.kraken({"enableRateLimit": True})
    binance = ccxt.binance({"enableRateLimit": True})
    k_markets = kraken.load_markets()
    b_markets = binance.load_markets()
    with KRAKEN_FILE.open("w") as f:
        json.dump(
            {
                k: {
                    "symbol": v.get("symbol"),
                    "id": v.get("id"),
                    "altname": v.get("info", {}).get("altname"),
                    "base": v.get("base"),
                    "quote": v.get("quote"),
                    "baseId": v.get("baseId"),
                    "quoteId": v.get("quoteId"),
                }
                for k, v in k_markets.items()
            },
            f,
        )
    with BINANCE_FILE.open("w") as f:
        json.dump(
            {
                k: {
                    "symbol": v.get("symbol"),
                    "base": v.get("base"),
                    "quote": v.get("quote"),
                }
                for k, v in b_markets.items()
            },
            f,
        )
    addlog("[CACHE] Refreshed Kraken & Binance market maps", verbose_int=1, verbose_state=verbose)


def load_pair_cache() -> dict:
    if not KRAKEN_FILE.exists() or not BINANCE_FILE.exists():
        refresh_pair_cache()
    with KRAKEN_FILE.open() as fk, BINANCE_FILE.open() as fb:
        return {"kraken": json.load(fk), "binance": json.load(fb)}


def resolve_ccxt_symbols(
    coin: str, fiat: str, cache: dict | None = None, verbose: int = 0
) -> dict:
    cache = cache or load_pair_cache()
    coin, fiat = coin.upper(), fiat.upper()
    for m in cache["kraken"].values():
        if m.get("base") == coin and m.get("quote") == fiat:
            kraken_name = m.get("symbol")
            kraken_pair = m.get("id")
            break
    else:
        target_alt = f"{coin}{fiat}"
        for m in cache["kraken"].values():
            if m.get("altname") == target_alt:
                kraken_name, kraken_pair = m.get("symbol"), m.get("id")
                break
        else:
            raise RuntimeError(f"Kraken symbol not found for {coin}/{fiat}")

    binance_name = None
    for m in cache["binance"].values():
        if m.get("base") == coin and m.get("quote") == fiat:
            binance_name = m.get("symbol")
            break
    if not binance_name:
        raise RuntimeError(f"Binance symbol not found for {coin}/{fiat}")

    addlog(
        f"[RESOLVE] {coin}/{fiat} → Kraken: {kraken_name} ({kraken_pair}), Binance: {binance_name}",
        verbose_int=1,
        verbose_state=verbose,
    )
    return {
        "kraken_name": kraken_name,
        "kraken_pair": kraken_pair,
        "binance_name": binance_name,
    }


def resolve_wallet_codes(
    coin: str, fiat: str, cache: dict | None = None, verbose: int = 0
) -> dict:
    cache = cache or load_pair_cache()
    coin, fiat = coin.upper(), fiat.upper()
    for m in cache["kraken"].values():
        if m.get("base") == coin and m.get("quote") == fiat:
            base_code = m.get("baseId")
            quote_code = m.get("quoteId")
            addlog(
                f"[RESOLVE] Wallet codes → base={base_code} quote={quote_code}",
                verbose_int=1,
                verbose_state=verbose,
            )
            return {"base_wallet_code": base_code, "quote_wallet_code": quote_code}
    raise RuntimeError(f"Kraken wallet codes not found for {coin}/{fiat}")


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
