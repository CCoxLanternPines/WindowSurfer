"""Rename legacy raw/tag files to asset-based filenames."""

from __future__ import annotations

from pathlib import Path

from systems.utils.config import load_settings, resolve_path
from systems.utils.symbols import resolve_asset, resolve_tag


def main() -> None:
    root = resolve_path("")
    settings = load_settings()
    raw_dir = root / "data" / "raw"
    ledger_dir = root / "data" / "ledgers"

    for ledger_cfg in settings.get("ledger_settings", {}).values():
        tag = resolve_tag(ledger_cfg)
        asset = resolve_asset(ledger_cfg)

        old_raw = raw_dir / f"{tag}.csv"
        new_raw = raw_dir / f"{asset}.csv"
        if old_raw.exists() and not new_raw.exists():
            print(f"Renaming {old_raw} -> {new_raw}")
            old_raw.rename(new_raw)

        old_ledger = ledger_dir / f"{tag}.json"
        new_ledger = ledger_dir / f"{asset}.json"
        if old_ledger.exists() and not new_ledger.exists():
            print(f"[NOTE] Consider renaming {old_ledger} -> {new_ledger}")


if __name__ == "__main__":
    main()
