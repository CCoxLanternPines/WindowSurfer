from __future__ import annotations

"""Simple in-memory ledger utilities for simulations."""

import json
from typing import Dict, List

from systems.utils.path import find_project_root


class RamLedger:
    """Minimal ledger keeping raw trade state."""

    def __init__(self) -> None:
        self.open_notes: List[Dict] = []
        self.closed_notes: List[Dict] = []
        self.deposits: List[Dict] = []
        self.withdrawals: List[Dict] = []
        self.metadata: Dict = {}

    def open_note(self, note: Dict) -> None:
        self.open_notes.append(note)

    def close_note(self, note: Dict) -> None:
        if note not in self.open_notes:
            return
        self.open_notes.remove(note)
        self.closed_notes.append(note)

    # Accessors ---------------------------------------------------------
    def get_open_notes(self) -> List[Dict]:
        return list(self.open_notes)

    def get_active_notes(self) -> List[Dict]:
        return self.get_open_notes()

    def get_closed_notes(self) -> List[Dict]:
        return list(self.closed_notes)

    def get_deposits(self) -> List[Dict]:
        return list(self.deposits)

    def get_withdrawals(self) -> List[Dict]:
        return list(self.withdrawals)

    def get_metadata(self) -> Dict:
        return dict(self.metadata)

    def get_summary(self) -> Dict:
        realised = sum(float(n.get("gain_usdt", 0)) for n in self.closed_notes)
        return {
            "open_notes": len(self.open_notes),
            "closed_notes": len(self.closed_notes),
            "realised_pnl": realised,
        }


def save_ledger(tag: str, ledger: RamLedger) -> None:
    """Persist raw ledger state to ``data/ledgers/<tag>.json``."""
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
