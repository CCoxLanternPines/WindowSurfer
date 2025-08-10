from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime


@dataclass
class Note:
    """Represents a single open position for a tunnel."""

    buy_price: float
    qty: float
    maturity_price: float
    timestamp: datetime
    partial_sold: bool = False


class Ledger:
    """Tracks coin sized notes for each symbol/tunnel pair."""

    def __init__(self, settings: Optional[Dict] = None) -> None:
        self.notes: Dict[str, Dict[str, List[Note]]] = {}
        self.settings = settings or {}
        self.min_order_fiat = float(self.settings.get("min_order_fiat", 0))
        self.min_order_coin = float(self.settings.get("min_order_coin", 0))
        self.partial_sell_midpoint = float(
            self.settings.get("partial_sell_midpoint", 0.5)
        )

    # ------------------------------------------------------------------
    def buy(
        self,
        symbol: str,
        tunnel_id: str,
        qty: float,
        price: float,
        maturity_price: float,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """Append a new note if order requirements are satisfied."""

        if qty < self.min_order_coin or price * qty < self.min_order_fiat:
            return False

        symbol_dict = self.notes.setdefault(symbol, {})
        tunnel_notes = symbol_dict.setdefault(tunnel_id, [])
        tunnel_notes.append(
            Note(
                buy_price=price,
                qty=qty,
                maturity_price=maturity_price,
                timestamp=timestamp or datetime.utcnow(),
            )
        )
        return True

    # ------------------------------------------------------------------
    def sell(
        self,
        symbol: str,
        tunnel_id: str,
        qty: float,
        price: float,
        note_idx: int = 0,
        partial: bool = False,
    ) -> float:
        """Remove or shrink a note. Returns fiat value of sale."""

        if qty < self.min_order_coin or price * qty < self.min_order_fiat:
            return 0.0

        symbol_dict = self.notes.get(symbol, {})
        notes = symbol_dict.get(tunnel_id, [])
        if note_idx >= len(notes):
            return 0.0
        note = notes[note_idx]
        sell_qty = min(qty, note.qty)
        note.qty -= sell_qty
        if partial:
            note.partial_sold = True
        if note.qty <= 0:
            del notes[note_idx]
        return sell_qty * price

    # ------------------------------------------------------------------
    def get_total_coin(self, symbol: str) -> float:
        """Sum quantity across all tunnels for symbol."""
        total = 0.0
        for tunnel_notes in self.notes.get(symbol, {}).values():
            total += sum(n.qty for n in tunnel_notes)
        return total

    # ------------------------------------------------------------------
    def get_fiat_value(self, symbol: str, price: float) -> float:
        """Convert total coin for symbol to fiat using price."""
        return self.get_total_coin(symbol) * price

    # ------------------------------------------------------------------
    def get_notes(self, symbol: str, tunnel_id: str) -> List[Note]:
        """Return list of notes for a tunnel."""
        return self.notes.get(symbol, {}).get(tunnel_id, [])

    # ------------------------------------------------------------------
    def total_fiat_value(self, prices: Dict[str, float]) -> float:
        """Total fiat value across all symbols using provided prices."""
        total = 0.0
        for sym, price in prices.items():
            total += self.get_fiat_value(sym, price)
        return total
