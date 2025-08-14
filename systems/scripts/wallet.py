from __future__ import annotations

"""Wallet helper functions."""

from systems.utils.addlog import addlog
from systems.utils.resolve_symbol import split_tag
from .ledger import init_ledger
from .kraken_utils import get_kraken_balance


def show_wallet(ledger: str | None, verbose: int = 0) -> None:
    """Display Kraken wallet balances for the given ledger."""

    ledger_cfg = init_ledger(ledger)
    _, quote_asset = split_tag(ledger_cfg["tag"])
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
