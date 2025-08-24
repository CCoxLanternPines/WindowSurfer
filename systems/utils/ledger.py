from __future__ import annotations
"""Simple JSON ledger utilities.

Schema::
    {
      "account": "Kris",
      "market": "DOGE/USD",
      "mode": "sim|live",
      "entries": [
        {
          "candle_idx": 123,
          "timestamp": 1692921600,
          "side": "BUY|SELL|PASS",
          "price": 0.0812,
          "size_usd": 250.0,
          "entry_price": 0.0810,
          "roi": 0.024,
          "pressure_buy": 3.0,
          "pressure_sell": 1.0,
          "features": {...}
        }
      ]
    }
"""

import json
import os
from typing import Any, Dict


def init_ledger(account: str, market: str, mode: str) -> Dict[str, Any]:
    """Return a new ledger dictionary."""
    return {"account": account, "market": market, "mode": mode, "entries": []}


def append_entry(ledger: Dict[str, Any], entry: Dict[str, Any]) -> None:
    """Append ``entry`` to ``ledger`` in-place."""
    ledger.setdefault("entries", []).append(entry)


def save_ledger(ledger: Dict[str, Any], path: str) -> None:
    """Persist ``ledger`` to ``path`` as JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(ledger, fh, indent=2)


def load_ledger(path: str) -> Dict[str, Any]:
    """Load ledger data from ``path`` if it exists."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}
