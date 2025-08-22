from __future__ import annotations
from typing import Dict, Any

def resolve_data_path(settings: Dict[str, Any], ledger: str) -> str:
    tag = settings["ledger_settings"][ledger]["tag"]
    return f"data/sim/{tag}_1h.csv"
