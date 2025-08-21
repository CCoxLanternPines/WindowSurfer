from __future__ import annotations

"""Minimal simulation engine using account and market configuration."""

from systems.scripts.runtime_state import build_state
from systems.utils.addlog import addlog


def run_simulation(
    *,
    account: str,
    market: str,
    verbose: int = 0,
    timeframe: str | None = None,
    viz: bool = False,
) -> None:
    """Run a basic simulation for ``account`` on ``market``."""
    state = build_state(account, market, mode="sim")
    addlog(
        f"[SIM] account={account} market={market} timeframe={timeframe}",
        verbose_int=0,
        verbose_state=True,
    )
    return
