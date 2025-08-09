"""In-memory ledger for Phase 0 simulation."""

from __future__ import annotations

from typing import List, Dict


class Ledger:
    """Tracks open and closed notes."""

    def __init__(self, maturity_gain: float) -> None:
        self.maturity_gain = maturity_gain
        self.open_notes: List[Dict[str, float]] = []
        self.closed_notes: List[Dict[str, float]] = []

    def buy(self, price: float, amount: float, step: int) -> None:
        self.open_notes.append(
            {"entry_price": price, "amount": amount, "entry_step": step}
        )

    def check_sells(self, current_price: float, step: int) -> float:
        """Check open notes for maturity and sell."""
        still_open: List[Dict[str, float]] = []
        proceeds = 0.0
        for note in self.open_notes:
            gain = (current_price - note["entry_price"]) / note["entry_price"]
            if gain >= self.maturity_gain:
                note.update(
                    {"exit_price": current_price, "exit_step": step, "gain": gain}
                )
                self.closed_notes.append(note)
                proceeds += note["amount"] * current_price
            else:
                still_open.append(note)
        self.open_notes = still_open
        return proceeds
