"""Utilities to load historical candle data without pandas."""

from __future__ import annotations

import csv
from typing import Any, Dict, List

from systems.utils.path import find_project_root


class _Column(list):
    """List subclass providing ``min`` and ``max`` like pandas Series."""

    def min(self) -> float:  # type: ignore[override]
        return float(min(self))

    def max(self) -> float:  # type: ignore[override]
        return float(max(self))


class _ILoc:
    """Helper to provide DataFrame-like ``iloc`` indexing."""

    def __init__(self, rows: List[Dict[str, Any]]):
        self._rows = rows

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return SimpleDataFrame(self._rows[idx])
        return self._rows[idx]


class SimpleDataFrame:
    """Minimal subset of pandas DataFrame used by the simulator."""

    def __init__(self, rows: List[Dict[str, Any]]):
        self._rows = rows
        self.iloc = _ILoc(self._rows)

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def empty(self) -> bool:
        return not self._rows

    def __getitem__(self, key: str) -> _Column:
        return _Column(row[key] for row in self._rows)


def fetch_candles(tag: str) -> SimpleDataFrame:
    """Load historical candles for ``tag`` from ``data/raw``."""

    root = find_project_root()
    path = root / "data" / "raw" / f"{tag.upper()}.csv"
    if not path.exists():  # pragma: no cover - file presence check
        raise FileNotFoundError(f"Candle file not found: {path}")

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            row: Dict[str, Any] = {
                "timestamp": int(float(r.get("timestamp", 0))),
                "open": float(r.get("open", 0)),
                "high": float(r.get("high", 0)),
                "low": float(r.get("low", 0)),
                "close": float(r.get("close", 0)),
            }
            rows.append(row)

    return SimpleDataFrame(rows)

