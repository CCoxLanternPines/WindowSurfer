from __future__ import annotations

"""Prime a Kraken API data snapshot for reuse within the current hour."""

from systems.utils.addlog import addlog
from systems.utils.snapshot import prime_snapshot


def prime_kraken_snapshot(
    api_key: str, api_secret: str, ledger_name: str, verbose: int = 0
) -> None:
    """Fetch balance and trades once and cache them for the hour."""
    prime_snapshot(ledger_name, api_key, api_secret)
    addlog(
        f"[SNAPSHOT] Cached Kraken balance and trades for {ledger_name}",
        verbose_int=3,
        verbose_state=verbose,
    )
