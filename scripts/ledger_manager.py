from __future__ import annotations

"""Simple in-memory ledger utilities for simulations."""

import json
from typing import Dict, List

from systems.utils.path import find_project_root


class RamLedger:
    """Minimal ledger keeping open and closed trade notes."""

    def __init__(self) -> None:
        self.open_notes: List[Dict] = []
        self.closed_notes: List[Dict] = []
        self.pnl: float = 0.0

    def open_note(self, note: Dict) -> None:
        self.open_notes.append(note)

    def close_note(self, note: Dict) -> None:
        if note not in self.open_notes:
            return
        self.open_notes.remove(note)
        self.closed_notes.append(note)
        entry = float(note.get("entry_usdt", 0))
        exit_ = float(note.get("exit_usdt", 0))
        self.pnl += exit_ - entry

    def get_active_notes(self) -> List[Dict]:
        return list(self.open_notes)

    def get_summary(self) -> Dict:
        return {
            "open_notes": len(self.open_notes),
            "closed_notes": len(self.closed_notes),
            "realised_pnl": self.pnl,
        }


def save_ledger(ledger: RamLedger, capital: float) -> None:
    """Persist ledger state and remaining capital to tmp JSON file."""
    root = find_project_root()
    out_dir = root / "data" / "tmp"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ledgersimulation.json"
    print("[LEDGER] Saving ledger to:", out_path)
    try:
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "capital": capital,
                    "open_notes": ledger.open_notes,
                    "closed_notes": ledger.closed_notes,
                    "pnl": ledger.pnl,
                },
                f,
                indent=2,
            )
    except Exception as exc:
        print("[LEDGER] Failed to save ledger:", exc)
        print("[LEDGER] Ledger summary:", json.dumps(ledger.get_summary()))
