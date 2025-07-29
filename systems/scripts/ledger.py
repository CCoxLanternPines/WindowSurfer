"""Ledger base classes and simple RAM implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict
import uuid
import os
import json

class LedgerBase(ABC):
    """Abstract interface for ledger implementations."""

    @abstractmethod
    def open_note(self, note: Dict) -> None:
        """Add a new open note to the ledger."""
        raise NotImplementedError

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
        if "note_id" not in note:
            note["note_id"] = str(uuid.uuid4())
        self.open_notes.append(note)

    # Maintain backward compatibility with new API
    def open_note(self, note: Dict) -> None:
        """Alias for ``add_note`` to open a new note."""
        self.add_note(note)


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

        invested_total = sum(
            float(n.get("entry_usdt", 0)) for n in self.closed_notes + self.open_notes
        )

        return {
            "num_open": num_open,
            "num_closed": num_closed,
            "total_pnl_usdt": total_pnl_usdt,
            "total_gain_pct": total_gain_pct,
            "estimated_kraken_balance": estimated_balance,
            "total_invested_usdt": invested_total
        }
        
    def get_roi_per_month(self, candle_count: int, candle_minutes: int = 60) -> float:
        """
        Calculates ROI per month from closed trades.
        ROI = (total pnl / total invested) / months
        """
        if not candle_count:
            return 0.0

        months = max((candle_count * candle_minutes) / (30 * 24 * 60), 1)
        pnl = sum(n["exit_usdt"] - n["entry_usdt"] for n in self.closed_notes)
        invested = sum(n["entry_usdt"] for n in self.closed_notes)

        if invested == 0:
            return 0.0

        return pnl / invested / months

    def get_avg_gain_per_month(self, candle_count: int, candle_minutes: int = 60) -> float:
        """
        Returns average gain percent per month over the simulation duration.
        Based on average gain per trade, normalized across months.
        """
        if not candle_count:
            return 0.0

        months = max((candle_count * candle_minutes) / (30 * 24 * 60), 1)
        total_gain = sum(float(n.get("gain_pct", 0)) for n in self.closed_notes)
        avg_gain = total_gain / max(len(self.closed_notes), 1)
        return avg_gain / months

    def get_trade_counts_by_strategy(self) -> dict:
        """Count total and open trades per strategy."""
        counts = {}

        for note in self.closed_notes + self.open_notes:
            strategy = note.get("strategy", "unknown")
            if strategy not in counts:
                counts[strategy] = {"total": 0, "open": 0}
            counts[strategy]["total"] += 1

        for note in self.open_notes:
            strategy = note.get("strategy", "unknown")
            if strategy not in counts:
                counts[strategy] = {"total": 0, "open": 0}
            counts[strategy]["open"] += 1

        return counts


class FileLedger(RamLedger):
    """Ledger that persists all notes to a JSON file."""

    def __init__(self, path: str) -> None:
        self.path = path
        super().__init__()
        self._load()

    def _load(self) -> None:
        """Load existing ledger data from ``self.path`` if available."""
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.open_notes = data.get("open_notes", [])
            self.closed_notes = data.get("closed_notes", [])
            self.pnl = sum(
                float(n.get("exit_usdt", 0)) - float(n.get("entry_usdt", 0))
                for n in self.closed_notes
            )
        except Exception:
            # If loading fails just start with empty ledger
            self.open_notes = []
            self.closed_notes = []
            self.pnl = 0.0

    def _sync(self) -> None:
        """Write current ledger state to ``self.path``."""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({
                "open_notes": self.open_notes,
                "closed_notes": self.closed_notes
            }, f, indent=2)

    def open_note(self, note: Dict) -> None:
        super().open_note(note)
        self._sync()

    def close_note(self, note: Dict) -> None:
        super().close_note(note)
        self._sync()
