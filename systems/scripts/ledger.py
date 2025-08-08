from __future__ import annotations

"""Simple in-memory ledger for simulations and live trading."""

import json
from typing import Dict, List

from systems.utils.config import resolve_path


class Ledger:
    """Track raw trade state and compute summary metrics on demand."""

    def __init__(self) -> None:
        self.open_notes: List[Dict] = []
        self.closed_notes: List[Dict] = []
        self.metadata: Dict = {}

    # Basic note management -------------------------------------------------
    def append_open(self, note: Dict) -> None:
        """Register a newly opened note.

        Notes are stored as raw dictionaries and are expected to contain an
        ``id`` field that uniquely identifies them.  This mirrors the live
        ledger interface so that simulation and live code paths can operate on
        the same structures.
        """
        self.open_notes.append(note)

    # Backwards compatibility ------------------------------------------------
    def open_note(self, note: Dict) -> None:  # pragma: no cover - legacy alias
        """Legacy alias for :meth:`append_open`."""
        self.append_open(note)

    def close_note(self, note_or_id, close_payload: Dict | None = None) -> bool:
        """Close a note by id or by direct reference.

        ``close_payload`` is merged into the note before archiving.  This
        function accepts either a note ``dict`` (legacy behaviour) or an id.
        It returns ``True`` when a note was successfully closed.
        """

        if isinstance(note_or_id, dict):
            note = note_or_id
        else:
            note = next(
                (n for n in self.open_notes if n.get("id") == note_or_id), None
            )
            if note is None:
                return False
        if note not in self.open_notes:
            return False
        self.open_notes.remove(note)
        if close_payload:
            note.update(close_payload)
        self.closed_notes.append(note)
        return True

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
    def load_ledger(asset: str, *, sim: bool = False) -> "Ledger":
        """Load a ledger for ``asset`` depending on mode."""
        root = resolve_path("")
        ledger = Ledger()

        asset = asset.upper()

        if sim:
            path = root / "data" / "tmp" / "simulation" / f"{asset}.json"
            if path.exists():
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                ledger.open_notes = data.get("open_notes", [])
                ledger.closed_notes = data.get("closed_notes", [])
                ledger.metadata = data.get("metadata", {})
            return ledger

        path = root / "data" / "ledgers" / f"{asset}.json"
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            ledger.open_notes = data.get("open_notes", [])
            ledger.closed_notes = data.get("closed_notes", [])
            ledger.metadata = data.get("metadata", {})
        return ledger


def save_ledger(
    asset: str,
    ledger: "Ledger" | dict,
    *,
    sim: bool = False,
    final_tick: int | None = None,
    summary: dict | None = None,
) -> None:
    """Persist ``ledger`` data to the canonical ledger directory."""

    root = resolve_path("")
    asset = asset.upper()

    if sim:
        out_dir = root / "data" / "tmp" / "simulation"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{asset}.json"
    else:
        out_dir = root / "data" / "ledgers"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{asset}.json"

    if isinstance(ledger, Ledger):
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
    else:
        ledger_data = ledger

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(ledger_data, f, indent=2)
