from __future__ import annotations

"""Wallet helper functions."""

from systems.utils.addlog import addlog
from systems.utils.resolve_symbol import split_tag, resolve_symbols, to_tag
from systems.utils.load_config import load_config
from .kraken_utils import get_kraken_balance
import ccxt


def show_wallet(account: str, market: str | None, verbose: int = 0) -> None:
    """Display Kraken wallet balances for the given account and market."""

    cfg = load_config()
    acct_cfg = cfg.get("accounts", {}).get(account)
    if not acct_cfg:
        addlog(
            f"[ERROR] Unknown account {account}",
            verbose_int=1,
            verbose_state=verbose,
        )
        return
    markets = acct_cfg.get("markets", {})
    market_symbol = market or next(iter(markets.keys()), None)
    if not market_symbol or market_symbol not in markets:
        addlog(
            f"[ERROR] Market {market_symbol} not configured for account {account}",
            verbose_int=1,
            verbose_state=verbose,
        )
        return

    client = ccxt.kraken({"enableRateLimit": True})
    symbols = resolve_symbols(client, market_symbol)
    tag = to_tag(symbols["kraken_name"])
    _, quote_asset = split_tag(tag)
    balances = get_kraken_balance(quote_asset, verbose)

    if verbose >= 1:
        addlog("[WALLET] Kraken Balance", verbose_int=1, verbose_state=verbose)
        addlog(str(balances), verbose_int=2, verbose_state=verbose)
        for asset, amount in balances.items():
            val = float(amount)
            if val == 0:
                continue
            fmt = f"{val:.2f}" if val > 1 else f"{val:.6f}"
            if asset.upper() == quote_asset.upper():
                addlog(f"{asset}: ${fmt}", verbose_int=1, verbose_state=verbose)
            else:
                addlog(f"{asset}: {fmt}", verbose_int=1, verbose_state=verbose)
