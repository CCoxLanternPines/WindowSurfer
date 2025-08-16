from __future__ import annotations

"""Simple in-memory ledger for simulations and live trading."""

import json
from typing import Dict, List

from systems.utils.config import resolve_path, load_settings, load_ledger_config


class Ledger:
    """Track raw trade state and compute summary metrics on demand."""

    def __init__(self) -> None:
        self.open_notes: List[Dict] = []
        self.closed_notes: List[Dict] = []
        self.metadata: Dict = {}

    # Basic note management -------------------------------------------------
    def open_note(self, note: Dict) -> None:
        """Register a newly opened note.

        Notes may contain additional metadata such as:

        ``p_buy``            – window position at entry
        ``unlock_p``         – bounce threshold to re-enable buys
        ``target_price``     – price at which the note should be sold
        ``target_roi``       – expected ROI at target price
        ``window_name``      – name of the window configuration
        ``window_size``      – size of the evaluation window
        ``created_idx/ts``   – creation index or timestamp

        These fields are stored verbatim and are not interpreted by the
        ledger itself but are useful for downstream analysis and evaluation.
        """
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
    def load_ledger(
        ledger_name: str, *, tag: str | None = None, sim: bool = False
    ) -> "Ledger":
        """Load a ledger for ``ledger_name``.

        If a legacy ledger file named after ``tag`` exists, it will be
        migrated to the ``ledger_name`` convention on first load.
        """

        root = resolve_path("")
        ledger = Ledger()

        if sim:
            out_dir = root / "data" / "tmp" / "simulation"
        else:
            out_dir = root / "data" / "ledgers"

        out_path = out_dir / f"{ledger_name}.json"

        if tag and not out_path.exists():
            legacy_path = out_dir / f"{tag}.json"
            if legacy_path.exists():
                legacy_path.rename(out_path)

        if out_path.exists():
            with out_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            ledger.open_notes = data.get("open_notes", [])
            ledger.closed_notes = data.get("closed_notes", [])
            ledger.metadata = data.get("metadata", {})
        return ledger


def load_ledger(
    ledger_name: str, *, tag: str | None = None, sim: bool = False
) -> "Ledger":
    """Public wrapper to load a ledger from disk."""

    return Ledger.load_ledger(ledger_name, tag=tag, sim=sim)


def save_ledger(
    ledger_name: str,
    ledger: "Ledger" | dict,
    *,
    sim: bool = False,
    final_tick: int | None = None,
    summary: dict | None = None,
    tag: str | None = None,
) -> None:
    """Persist ``ledger`` data to the canonical ledger directory."""

    root = resolve_path("")

    if sim:
        out_dir = root / "data" / "tmp" / "simulation"
    else:
        out_dir = root / "data" / "ledgers"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ledger_name}.json"

    if tag and not out_path.exists():
        legacy_path = out_dir / f"{tag}.json"
        if legacy_path.exists():
            legacy_path.rename(out_path)

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

def init_ledger(
    ledger_name: str,
    *,
    tag: str | None = None,
    sim: bool = False,
) -> Ledger:
    """Load ``ledger_name`` and ensure its file exists on disk."""

    ledger = load_ledger(ledger_name, tag=tag, sim=sim)
    save_ledger(ledger_name, ledger, tag=tag, sim=sim)
    return ledger


def resolve_ledger_config(ledger_name: str | None) -> Dict:
    """Return configuration dictionary for ``ledger_name``.

    If ``ledger_name`` is ``None``, the first ledger defined in settings is
    returned. Mirrors previous inline logic previously in ``bot.py``.
    """

    settings = load_settings()
    if ledger_name:
        return load_ledger_config(ledger_name)
    return next(iter(settings.get("ledger_settings", {}).values()))
