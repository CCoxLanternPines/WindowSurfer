from __future__ import annotations

"""Helpers for recording per-candle decisions during live runs."""

import json
from pathlib import Path
from typing import Any, List


def write_action(
    idx: int,
    row: Any,
    sells: int,
    buys: int,
    notes: List[dict],
    cap: float,
    account: str,
    coin: str,
) -> None:
    """Append an action record for the given candle."""

    path = Path("data") / "actions" / f"{account}_{coin}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "idx": int(idx),
        "timestamp": int(row.get("timestamp", 0)),
        "sells": int(sells),
        "buys": int(buys),
        "open_notes": len(notes),
        "capital": float(cap),
    }
    with path.open("a", encoding="utf-8") as f:
        json.dump(record, f)
        f.write("\n")


__all__ = ["write_action"]

