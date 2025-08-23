from __future__ import annotations

import json
from pathlib import Path

from .rules import buy_decision, sell_decision


WEIGHTS_PATH = Path(__file__).with_name("weights.json")
try:
    with open(WEIGHTS_PATH) as fh:
        WEIGHTS = json.load(fh)
except Exception:
    WEIGHTS = {}


def normalize(key: str, val: float | int | None) -> float:
    try:
        return float(val) / 100.0
    except Exception:
        return 0.0


def weighted_score(features: dict, weights: dict) -> float:
    score = 0.0
    for key, w in weights.items():
        if not w:
            continue
        val = features.get(key)
        if val is not None:
            score += w * normalize(key, val)
    return score


def run_arbiter(
    features: dict,
    position_state: str,
    debug: bool = False,
    return_score: bool = False,
):
    """Return a decision, reasons, and optionally the weighted score."""

    score = weighted_score(features, WEIGHTS)
    reasons: list[str] = []

    decision = "HOLD"
    if position_state == "flat":
        buy, why = buy_decision(features, debug=debug)
        reasons.extend(why)
        if buy:
            decision = "BUY"
    elif position_state == "long":
        sell, why = sell_decision(features, debug=debug)
        reasons.extend(why)
        if sell:
            decision = "SELL"

    if return_score:
        return decision, reasons, score
    return decision, reasons

