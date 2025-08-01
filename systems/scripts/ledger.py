from __future__ import annotations

"""Simple in-memory ledger for simulations."""

from typing import Dict, List


class Ledger:
    """Track open/closed notes, realised PnL and idle capital."""

    def __init__(self, capital: float = 0.0) -> None:
        self.capital: float = capital
        self.open_notes: List[Dict] = []
        self.closed_notes: List[Dict] = []
        self.pnl: float = 0.0

    # Basic note management -------------------------------------------------
    def open_note(self, note: Dict) -> None:
        """Register a newly opened note."""
        self.open_notes.append(note)

    def close_note(self, note: Dict) -> None:
        """Move ``note`` from open to closed and update realised PnL."""
        if note not in self.open_notes:
            return
        self.open_notes.remove(note)
        self.closed_notes.append(note)
        entry = float(note.get("entry_usdt", 0))
        exit_ = float(note.get("exit_usdt", 0))
        self.pnl += exit_ - entry

    # Capital helpers -------------------------------------------------------
    def set_capital(self, value: float) -> None:
        self.capital = value

    def get_capital(self) -> float:
        return self.capital

    # Accessors -------------------------------------------------------------
    def get_open_notes(self) -> List[Dict]:
        return list(self.open_notes)

    def get_active_notes(self) -> List[Dict]:
        return self.get_open_notes()

    def get_closed_notes(self) -> List[Dict]:
        return list(self.closed_notes)

    # Summary ---------------------------------------------------------------
    def get_account_summary(self, starting_capital: float) -> dict:
        realised_pnl = sum(n.get("gain_usdt", 0) for n in self.get_closed_notes())
        idle_capital = self.get_capital()
        open_value = sum(
            n.get("entry_amount", 0) * n.get("entry_price", 0) for n in self.get_open_notes()
        )
        ending_value = idle_capital + open_value + realised_pnl
        net_gain = ending_value - starting_capital
        roi = (net_gain / starting_capital) * 100 if starting_capital else 0.0

        return {
            "starting_capital": round(starting_capital, 2),
            "realised_pnl": round(realised_pnl, 2),
            "idle_capital": round(idle_capital, 2),
            "open_note_value": round(open_value, 2),
            "ending_value": round(ending_value, 2),
            "net_gain": round(net_gain, 2),
            "roi_pct": round(roi, 2),
            "closed_notes": len(self.get_closed_notes()),
            "open_notes": len(self.get_open_notes()),
            "total_notes": len(self.get_closed_notes()) + len(self.get_open_notes()),
        }
