from __future__ import annotations

"""Simple in-memory ledger for simulations."""

import json
from typing import Dict, List

from systems.utils.path import find_project_root


class Ledger:
    """Track raw trade state and compute summary metrics on demand."""

    def __init__(self) -> None:
        self.open_notes: List[Dict] = []
        self.closed_notes: List[Dict] = []
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

    def get_total_liquid_value(self, final_price: float) -> float:
        """Return all value assuming open notes liquidate at ``final_price``."""
        open_value = sum(
            n.get("entry_amount", 0) for n in self.get_open_notes()
        ) * final_price
        realised = sum(
            n.get("entry_amount", 0)
            * (n.get("exit_price", 0) - n.get("entry_price", 0))
            for n in self.get_closed_notes()
        )
        return open_value + realised

    # Summary ---------------------------------------------------------------
    def get_account_summary(self, final_price: float) -> dict:
        open_amount = sum(n.get("entry_amount", 0) for n in self.get_open_notes())
        open_value = open_amount * final_price
        realised = sum(
            n.get("entry_amount", 0)
            * (n.get("exit_price", 0) - n.get("entry_price", 0))
            for n in self.get_closed_notes()
        )
        total_value = open_value + realised

        return {
            "final_price": round(final_price, 8),
            "open_coin_amount": round(open_amount, 8),
            "open_value": round(open_value, 2),
            "realized_gain": round(realised, 2),
            "total_value": round(total_value, 2),
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
            ledger.metadata = data.get("metadata", {})
        return ledger

    @staticmethod
    def save_ledger(
        tag: str,
        ledger: "Ledger",
        *,
        final_tick: int | None = None,
        summary: dict | None = None,
    ) -> None:
        """Persist ``ledger`` to ``data/ledgers/<tag>.json``."""
        root = find_project_root()
        out_dir = root / "data" / "ledgers"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{tag}.json"

        ledger_data = {
            "open_notes": ledger.get_open_notes(),
            "closed_notes": ledger.get_closed_notes(),
        }

        if final_tick is not None:
            ledger_data["final_tick"] = final_tick

        if summary:
            ledger_data["closed_notes_count"] = summary.get("closed_notes")
            ledger_data["open_notes_count"] = summary.get("open_notes")
            ledger_data["realized_gain"] = summary.get("realized_gain")
            ledger_data["final_value"] = summary.get("total_value")

        metadata = ledger.get_metadata()
        if metadata:
            ledger_data["metadata"] = metadata

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(ledger_data, f, indent=2)
