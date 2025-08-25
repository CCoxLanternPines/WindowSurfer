from __future__ import annotations

"""Lightweight in-memory book for open notes during testing."""

from typing import Dict, List


class AccountBook:
    """Track open and closed notes for account validation."""

    def __init__(self) -> None:
        self.open_notes: List[Dict] = []
        self.closed_notes: List[Dict] = []

    def open_note(self, note: Dict) -> None:
        """Register a new open note."""
        self.open_notes.append(note)

    def close_note(self, note: Dict) -> None:
        """Move ``note`` from open to closed."""
        if note in self.open_notes:
            self.open_notes.remove(note)
            self.closed_notes.append(note)

    def get_open_notes(self) -> List[Dict]:
        """Return a copy of open notes."""
        return list(self.open_notes)
