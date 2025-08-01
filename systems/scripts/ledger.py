from __future__ import annotations

"""Simple in-memory ledger for simulations."""

import json
from typing import Dict, List

from systems.utils.path import find_project_root


class Ledger:
    """Track raw trade state and compute summary metrics on demand."""

    def __init__(self, capital: float = 0.0) -> None:
        self.capital: float = capital
        self.open_notes: List[Dict] = []
        self.closed_notes: List[Dict] = []
        self.deposits: List[Dict] = []
        self.withdrawals: List[Dict] = []
        self.metadata: Dict = {}

    # Basic note management -------------------------------------------------
    def open_note(self, note: Dict) -> None:
        """Register a newly opened note."""
        self.open_notes.append(note)

    def close_note(self, note: Dict) -> None:
        """Move ``note`` from open to closed."""
        if note not in self.open_notes:
            return
        self.open_notes.remove(note)
        self.closed_notes.append(note)

    # Capital helpers -------------------------------------------------------
    def set_capital(self, value: float) -> None:
        self.capital = value

    def get_capital(self) -> float:
        return self.capital

    # Deposit/withdraw helpers ---------------------------------------------
    def add_deposit(self, amount: float) -> None:
        self.deposits.append({"amount": float(amount)})
        self.capital += float(amount)

    def add_withdrawal(self, amount: float) -> None:
        self.withdrawals.append({"amount": float(amount)})
        self.capital -= float(amount)

    def get_deposits(self) -> List[Dict]:
        return list(self.deposits)

    def get_withdrawals(self) -> List[Dict]:
        return list(self.withdrawals)

    def set_metadata(self, metadata: Dict) -> None:
        self.metadata = metadata

    def get_metadata(self) -> Dict:
        return dict(self.metadata)

    # Accessors -------------------------------------------------------------
    def get_open_notes(self) -> List[Dict]:
        return list(self.open_notes)

    def get_active_notes(self) -> List[Dict]:
        return self.get_open_notes()

    def get_closed_notes(self) -> List[Dict]:
        return list(self.closed_notes)

    def get_total_liquid_value(self) -> float:
        """Return all capital that could be withdrawn immediately."""
        realised = sum(n.get("gain_usdt", 0) for n in self.get_closed_notes())
        open_value = sum(
            n.get("entry_amount", 0) * n.get("entry_price", 0)
            for n in self.get_open_notes()
        )
        idle_capital = self.get_capital()
        return realised + open_value + idle_capital

    # Summary ---------------------------------------------------------------
    def get_account_summary(self, starting_capital: float) -> dict:
        realised = sum(n.get("gain_usdt", 0) for n in self.get_closed_notes())
        idle_capital = self.get_capital()
        open_value = sum(
            n.get("entry_amount", 0) * n.get("entry_price", 0)
            for n in self.get_open_notes()
        )
        total_value = realised + open_value + idle_capital
        net_gain = total_value - starting_capital
        roi = (net_gain / starting_capital) * 100 if starting_capital else 0.0

        return {
            "starting_capital": round(starting_capital, 2),
            "realised_gain": round(realised, 2),
            "open_value": round(open_value, 2),
            "idle_capital": round(idle_capital, 2),
            "total_value": round(total_value, 2),
            "net_gain": round(net_gain, 2),
            "roi": round(roi, 2),
            "closed_notes": len(self.get_closed_notes()),
            "open_notes": len(self.get_open_notes()),
        }

    # Persistence -----------------------------------------------------------
    @staticmethod
    def load_ledger(tag: str) -> "Ledger":
        """Load a ledger from ``data/ledgers/<tag>.json`` if it exists."""
        root = find_project_root()
        path = root / "data" / "ledgers" / f"{tag}.json"
        ledger = Ledger()
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            ledger.open_notes = data.get("open_notes", [])
            ledger.closed_notes = data.get("closed_notes", [])
            ledger.deposits = data.get("deposits", [])
            ledger.withdrawals = data.get("withdrawals", [])
            ledger.metadata = data.get("metadata", {})
        return ledger

    @staticmethod
    def save_ledger(tag: str, ledger: "Ledger") -> None:
        """Persist ``ledger`` to ``data/ledgers/<tag>.json``."""
        root = find_project_root()
        out_dir = root / "data" / "ledgers"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{tag}.json"

        ledger_data = {
            "open_notes": ledger.get_open_notes(),
            "closed_notes": ledger.get_closed_notes(),
            "deposits": ledger.get_deposits(),
            "withdrawals": ledger.get_withdrawals(),
        }
        metadata = ledger.get_metadata()
        if metadata:
            ledger_data["metadata"] = metadata

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(ledger_data, f, indent=2)
