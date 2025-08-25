"""Lightweight wrappers around :mod:`graph` plotting helpers."""

from __future__ import annotations

from pathlib import Path

from graph import plot


def plot_from_json(sim_path: str) -> None:
    """Plot simulation results from a JSON ledger file.

    The JSON file is expected to contain ``candles_path`` pointing at the
    CSV of candle data. This is a very small stub to preserve the public
    API used by :mod:`sim`.
    """

    ledger_file = Path(sim_path)
    if not ledger_file.exists():  # pragma: no cover - best effort
        return
    import json

    with ledger_file.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    candles = data.get("meta", {}).get("candles_path")
    if candles:
        plot(str(sim_path), str(candles))


def plot_trades_from_ledger(account: str, market: str, mode: str = "live") -> None:
    """Plot trades from a ledger file for ``account`` and ``market``.

    This stub attempts to locate a ledger under ``data/{mode}`` and plot it
    alongside the corresponding candle data if available. It is intended as
    a minimal replacement for the richer legacy plotting utilities.
    """

    ledger = Path("data") / mode / account / f"{market}_ledger.json"
    candles = Path("data") / "candles" / mode / f"{market}.csv"
    if ledger.exists() and candles.exists():  # pragma: no cover - best effort
        plot(str(ledger), str(candles))

