"""Ledger base classes and simple RAM implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict


class LedgerBase(ABC):
    """Abstract interface for ledger implementations."""

    @abstractmethod
    def get_active_notes(self) -> List[Dict]:
        """Return a list of currently open notes"""
        raise NotImplementedError

    @abstractmethod
    def get_closed_notes(self) -> List[Dict]:
        """Return a list of all closed (exited) notes"""
        raise NotImplementedError

    @abstractmethod
    def get_summary(self) -> Dict:
        """Return a JSON-like summary for sync or visual inspection."""
        raise NotImplementedError


class RamLedger(LedgerBase):
    """In-memory ledger used for simulations and testing."""

    def __init__(self) -> None:
        self.open_notes: List[Dict] = []
        self.closed_notes: List[Dict] = []
        # Cumulative realised PnL in USDT
        self.pnl: float = 0.0

    # Convenience helpers -------------------------------------------------
    def add_note(self, note: Dict) -> None:
        """Add a new open note to the ledger."""
        self.open_notes.append(note)

    def close_note(self, note: Dict) -> None:
        """Move ``note`` from open to closed and update PnL."""
        if note not in self.open_notes:
            return
        self.open_notes.remove(note)
        self.closed_notes.append(note)
        entry_usdt = note.get("entry_usdt")
        exit_usdt = note.get("exit_usdt")
        if entry_usdt is not None and exit_usdt is not None:
            self.pnl += float(exit_usdt) - float(entry_usdt)

    # LedgerBase API ------------------------------------------------------
    def get_active_notes(self) -> List[Dict]:
        return list(self.open_notes)

    def get_closed_notes(self) -> List[Dict]:
        return list(self.closed_notes)

    def get_summary(self) -> Dict:
        num_open = len(self.open_notes)
        num_closed = len(self.closed_notes)

        # Aggregate pnl and gain percent from closed notes
        total_pnl_usdt = self.pnl
        gain_sum = 0.0
        gain_count = 0
        for note in self.closed_notes:
            pct = note.get("gain_pct")
            if pct is not None:
                gain_sum += float(pct)
                gain_count += 1
        total_gain_pct = gain_sum if gain_count == 0 else gain_sum / gain_count

        # Estimate Kraken balance assuming open notes close at entry_usdt
        closed_balance = sum(float(n.get("exit_usdt", 0)) for n in self.closed_notes)
        open_balance = sum(float(n.get("entry_usdt", 0)) for n in self.open_notes)
        estimated_balance = closed_balance + open_balance

        return {
            "num_open": num_open,
            "num_closed": num_closed,
            "total_pnl_usdt": total_pnl_usdt,
            "total_gain_pct": total_gain_pct,
            "estimated_kraken_balance": estimated_balance,
        }
