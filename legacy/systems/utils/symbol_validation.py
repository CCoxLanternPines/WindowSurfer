from __future__ import annotations

"""Utilities for validating exchange symbols and updating settings.json."""

from typing import Any, Dict, List
import json
import ccxt

from systems.utils.addlog import addlog
from systems.utils.config import resolve_path, load_settings
from systems.utils.symbols import resolve_asset


def _candidate_quotes(quote: str) -> List[str]:
    """Return possible quote currencies to try for a given ``quote``."""
    quote = quote.upper()
    candidates = [quote]
    if quote != "USD":
        candidates.append("USD")
    if quote != "USDT":
        candidates.append("USDT")
    # Remove duplicates while preserving order
    return list(dict.fromkeys(candidates))


def validate_exchange_symbols(settings: Dict[str, Any], ledger: str, tag: str) -> bool:
    """Validate and auto-correct exchange symbols for ``ledger``.

    Parameters
    ----------
    settings:
        Full settings dictionary loaded from ``settings.json``.
    ledger:
        Ledger key within ``settings['ledger_settings']`` to validate.
    tag:
        Trading pair tag, e.g. ``SOLUSD``.

    Returns
    -------
    bool
        ``True`` if symbols are valid or successfully corrected. ``False`` if
        no valid market could be found.
    """

    ledger_cfg = settings["ledger_settings"][ledger]
    base = resolve_asset(ledger_cfg)
    quote = tag.upper().replace(base, "", 1)
    quotes = _candidate_quotes(quote)

    changed = False

    # Kraken validation
    kraken_symbol = ledger_cfg.get("kraken_name")
    if kraken_symbol:
        exchange = ccxt.kraken({"enableRateLimit": True})
        markets = exchange.fetch_markets()
        market_symbols = {m["symbol"] for m in markets}
        if kraken_symbol not in market_symbols:
            new_symbol = None
            for q in quotes:
                candidate = f"{base}/{q}"
                if candidate in market_symbols:
                    new_symbol = candidate
                    break
            if new_symbol:
                addlog(
                    f"[FIX] Updated Kraken symbol from invalid '{kraken_symbol}' to '{new_symbol}'",
                    verbose_int=1,
                    verbose_state=True,
                )
                ledger_cfg["kraken_name"] = new_symbol
                changed = True
            else:
                return False

    # Binance validation
    binance_symbol = ledger_cfg.get("binance_name")
    if binance_symbol:
        exchange = ccxt.binance({"enableRateLimit": True})
        markets = exchange.fetch_markets()
        market_symbols = {m["symbol"] for m in markets}
        current = f"{base}/{binance_symbol[len(base):]}"
        if current not in market_symbols:
            new_symbol = None
            for q in quotes:
                candidate = f"{base}/{q}"
                if candidate in market_symbols:
                    new_symbol = candidate
                    break
            if new_symbol:
                fixed = new_symbol.replace("/", "")
                addlog(
                    f"[FIX] Updated Binance symbol from invalid '{binance_symbol}' to '{fixed}'",
                    verbose_int=1,
                    verbose_state=True,
                )
                ledger_cfg["binance_name"] = fixed
                changed = True
            else:
                return False

    if changed:
        settings_path = resolve_path("settings/settings.json")
        with settings_path.open("w", encoding="utf-8") as fh:
            json.dump(settings, fh, indent=2)
        load_settings(reload=True)

    return True


__all__ = ["validate_exchange_symbols"]
