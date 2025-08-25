from __future__ import annotations

"""Stub sell evaluator for discovery simulation."""

from typing import Any, Dict


def evaluate_sell(candle: Dict[str, Any], state: Dict[str, Any]) -> bool:
    """Return ``False`` to indicate no sell action."""
    return False
