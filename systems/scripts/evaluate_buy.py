from __future__ import annotations

"""Stub buy evaluator for discovery simulation."""

from typing import Any, Dict


def evaluate_buy(candle: Dict[str, Any], state: Dict[str, Any]) -> bool:
    """Return ``False`` to indicate no buy action."""
    return False
