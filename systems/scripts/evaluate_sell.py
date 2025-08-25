from __future__ import annotations

"""Evaluate whether any open positions should be closed."""

from typing import Any, Dict, List, Tuple


def evaluate_sell(
    idx: int,
    price: float,
    open_notes: List[Dict[str, float]],
    capital: float,
) -> Tuple[List[Dict[str, Any]], float, List[Dict[str, float]]]:
    """Return closed trade records and updated portfolio state."""
    trades_closed: List[Dict[str, Any]] = []
    updated_notes: List[Dict[str, float]] = []
    updated_capital = capital

    for note in open_notes:
        if price >= note["sell_price"]:
            sell_usd = note["units"] * price
            updated_capital += sell_usd
            trades_closed.append({"idx": idx, "price": price, "side": "SELL", "usd": sell_usd})
            print(
                f"SELL @ idx={idx}, entry={note['entry_price']:.2f}, target={note['sell_price']:.2f}, price={price:.2f}"
            )
        else:
            updated_notes.append(note)

    return trades_closed, updated_capital, updated_notes
