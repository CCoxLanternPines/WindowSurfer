from __future__ import annotations

"""Populate ``maturity_price`` for open notes lacking the field.

This migration reads all ledger files under ``data/ledgers`` and, for each
open note without a baked maturity target, writes a conservative value based on
``required_min_roi``.
"""

import json

from systems.scripts.ledger import Ledger, save_ledger
from systems.utils.config import resolve_path


def load_settings() -> dict:
    settings_path = resolve_path("settings/settings.json")
    with settings_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    root = resolve_path("")
    settings = load_settings()

    # Build lookup of required_min_roi per tag/window
    roi_lookup: dict[str, dict[str, float]] = {}
    for ledger_cfg in settings.get("ledger_settings", {}).values():
        tag = ledger_cfg.get("tag")
        window_map = {}
        for name, cfg in ledger_cfg.get("window_settings", {}).items():
            window_map[name] = cfg.get("required_min_roi", 0.0)
        roi_lookup[tag] = window_map

    ledgers_dir = root / "data" / "ledgers"
    if not ledgers_dir.exists():
        return

    for ledger_file in ledgers_dir.glob("*.json"):
        tag = ledger_file.stem
        ledger = Ledger.load_ledger(tag)
        changed = False
        for note in ledger.get_open_notes():
            if "maturity_price" in note:
                continue
            window = note.get("window")
            required_roi = roi_lookup.get(tag, {}).get(window, 0.0)
            entry_price = note.get("entry_price", 0.0)
            maturity_price = max(entry_price * (1 + required_roi), entry_price)
            note["maturity_price"] = maturity_price
            print(
                f"[BACKFILL] {tag} | {window} | entry={entry_price} -> maturity={maturity_price}"
            )
            changed = True
        if changed:
            save_ledger(tag, ledger)


if __name__ == "__main__":
    main()
