from __future__ import annotations

"""Utilities for persisting trade records to disk."""

import json
from pathlib import Path
from typing import Dict, Any


def write_trade(trade: Dict[str, Any] | None, account: str, coin: str) -> None:
    """Append ``trade`` to the ledger for ``account``/``coin``."""

    if not trade:
        return
    path = Path("data") / "ledgers" / f"{account}_{coin}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        json.dump(trade, f)
        f.write("\n")


__all__ = ["write_trade"]

