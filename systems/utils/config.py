import json
from pathlib import Path


def load_ledgers(path: str = "settings/ledgers.json") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_ledger_cfg(ledger_name: str, all_cfg: dict) -> dict:
    base = dict(all_cfg.get("default", {}))
    override = dict(all_cfg.get("ledgers", {}).get(ledger_name, {}))
    # Shallow merge is fine given current schema
    base.update(override)
    return base
