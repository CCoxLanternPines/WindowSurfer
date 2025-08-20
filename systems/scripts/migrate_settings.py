from __future__ import annotations

"""One-time helper to migrate legacy settings.json into split files."""

import json
from pathlib import Path

from systems.utils.config import resolve_path


def migrate_settings() -> None:
    settings_path = resolve_path("settings/settings.json")
    if not settings_path.exists():
        print("[MIGRATE][ERROR] settings/settings.json not found")
        return
    with settings_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    accounts = data.get("accounts")
    if accounts:
        out_path = resolve_path("settings/account_settings.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump({"accounts": accounts}, fh, indent=2)
        print(f"[MIGRATE] wrote {out_path}")

    coin_settings = data.get("coin_settings")
    if coin_settings:
        out_path = resolve_path("settings/coin_settings.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump({"coin_settings": coin_settings}, fh, indent=2)
        print(f"[MIGRATE] wrote {out_path}")


if __name__ == "__main__":
    migrate_settings()
