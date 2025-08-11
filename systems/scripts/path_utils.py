from pathlib import Path


def ledger_settings_path(name: str) -> Path:
    return Path("ledger/settings") / f"{name}.json"


def runtime_ledger_state_path(name: str, mode: str) -> Path:
    return Path("data/ledgers") / f"{name}_{mode}.json"
