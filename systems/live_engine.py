from __future__ import annotations

"""Minimal live engine using account and market configuration."""

from systems.scripts.runtime_state import build_state
from systems.utils.addlog import addlog


def run_live(*, account: str, market: str, dry: bool, verbose: int) -> None:
    """Run a live iteration for ``account`` on ``market``."""
    state = build_state(account, market, mode="live")
    addlog(f"[LIVE] account={account} market={market}", verbose_int=0, verbose_state=True)
    if dry:
        addlog("[LIVE] dry run completed", verbose_int=0, verbose_state=True)
    return
