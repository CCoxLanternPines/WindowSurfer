from __future__ import annotations

"""Shared candle iteration logic for sim and live engines."""

from typing import Any, Callable, Dict, Iterable, List, Tuple

import pandas as pd

HandlerFn = Callable[..., Any]
HandlersDict = Dict[str, HandlerFn]


def run_candle_loop(
    candles: pd.DataFrame,
    handlers: Dict[str, HandlerFn],
    account: str,
    coin: str,
) -> Tuple[float, List[Dict[str, float]]]:
    """Iterate over ``candles`` dispatching to strategy handlers.

    Parameters
    ----------
    candles:
        DataFrame of candles sorted in ascending order. Required columns:
        ``close`` and ``timestamp``.
    handlers:
        Mapping containing at minimum ``buy``, ``sell``, ``ledger`` and
        ``action`` callables. Optional ``on_candle`` callable receives
        ``(idx, row)`` each step. ``capital`` and ``open_notes`` keys may be
        provided to seed initial state.
    account:
        Account name, forwarded to ledger and action handlers.
    coin:
        Coin symbol, forwarded to ledger and action handlers.

    Returns
    -------
    Tuple[float, List[Dict[str, float]]]
        Final capital and list of open notes after processing all candles.
    """

    buy_fn: HandlerFn = handlers["buy"]
    sell_fn: HandlerFn = handlers["sell"]
    ledger_fn: HandlerFn = handlers["ledger"]
    action_fn: HandlerFn = handlers["action"]
    on_candle: HandlerFn | None = handlers.get("on_candle")

    capital: float = float(handlers.get("capital", 0.0))
    open_notes: List[Dict[str, float]] = list(handlers.get("open_notes", []))

    for idx, row in candles.iterrows():
        if on_candle:
            on_candle(idx, row)

        price = float(row.get("close", 0.0))

        closed, capital, open_notes = sell_fn(idx, row, open_notes, capital)
        for trade in closed:
            ledger_fn(trade, account, coin)
            action_fn(trade.get("side", "SELL"), idx, row, account, coin)

        trade, capital, open_notes = buy_fn(idx, row, open_notes, capital)
        if trade:
            ledger_fn(trade, account, coin)
            action_fn(trade.get("side", "BUY"), idx, row, account, coin)
        elif not closed:
            action_fn("HOLD", idx, row, account, coin)

    return capital, open_notes


__all__ = ["run_candle_loop"]
