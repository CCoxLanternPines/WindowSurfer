from __future__ import annotations

from typing import Any, Dict, Optional

from systems.scripts.execution_handler import load_or_fetch_snapshot
from systems.utils.resolve_symbol import load_pair_cache, resolve_wallet_codes


def build_runtime_state(
    settings: Dict[str, Any],
    ledger_cfg: Dict[str, Any],
    mode: str,
    *,
    ledger_name: str | None = None,
    prev: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build and refresh runtime state for engines.

    Parameters
    ----------
    settings:
        Global settings dictionary.
    ledger_cfg:
        Ledger configuration mapping containing ``coin`` and ``fiat``.
    mode:
        Either ``"sim"`` or ``"live"``.
    prev:
        Previous runtime state to carry over verbose level and unlock map.
    """

    prev = prev or {}
    general = settings.get("general_settings", {})
    limits = {
        "min_note_size": float(general.get("minimum_note_size", 0.0)),
        "max_note_usdt": float(general.get("max_note_usdt", float("inf"))),
    }

    buy_unlock_p = prev.get("buy_unlock_p", {})
    verbose = prev.get("verbose", 0)

    if mode == "sim":
        capital = float(settings.get("simulation_capital", 0.0))
    elif mode == "live":
        if not ledger_name:
            raise ValueError("ledger_name required for live mode")
        snapshot = load_or_fetch_snapshot(ledger_name)
        coin = ledger_cfg["coin"]
        fiat = ledger_cfg["fiat"]
        cache = load_pair_cache()
        codes = resolve_wallet_codes(coin, fiat, cache, prev.get("verbose", 0))
        balance = snapshot.get("balance", {})
        capital = float(balance.get(codes["quote_wallet_code"], 0.0))
    else:
        capital = prev.get("capital", 0.0)

    return {
        "capital": capital,
        "buy_unlock_p": buy_unlock_p,
        "verbose": verbose,
        "limits": limits,
    }
