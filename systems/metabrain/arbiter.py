from __future__ import annotations

import json
from pathlib import Path

from .rules import buy_decision, sell_decision

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_PATH = ROOT / "settings" / "weights.json"
try:
    with open(WEIGHTS_PATH) as fh:
        _WEIGHTS = json.load(fh)
except Exception:
    _WEIGHTS = {}

_DEFAULTS = _WEIGHTS.get("defaults", {})
_FEATURE_WEIGHTS = _WEIGHTS.get("features", {})


def compute_weighted_score(features: dict) -> tuple[float, list[tuple[str, float, float, str, float]]]:
    regime = features.get("regime_key", "")
    score = 0.0
    contributions: list[tuple[str, float, float, str, float]] = []
    for name, val in features.items():
        if name in ("regime", "regime_key"):
            continue
        if not isinstance(val, (int, float)):
            continue
        weight_info = _FEATURE_WEIGHTS.get(name, {})
        weight = 0.0
        regime_used = "unused"
        if isinstance(weight_info, dict):
            if regime in weight_info:
                weight = weight_info[regime]
                regime_used = regime
            elif "global" in weight_info:
                weight = weight_info["global"]
                regime_used = "global"
        contrib = val * weight
        contributions.append((name, val, weight, regime_used, contrib))
        score += contrib
    return score, contributions


def run_arbiter(
    features: dict,
    position_state: str,
    debug: bool = False,
    return_score: bool = False,
):
    """Return a decision, reasons, and optionally the weighted score."""

    score, contribs = compute_weighted_score(features)
    t_buy = float(_DEFAULTS.get("T_buy", 0.0))
    t_sell = float(_DEFAULTS.get("T_sell", 0.0))

    decision_from_score = "HOLD"
    if score >= t_buy:
        decision_from_score = "BUY"
    elif score <= -t_sell:
        decision_from_score = "SELL"

    reasons: list[str] = []
    guard_decision: str | None = None
    if position_state == "flat":
        buy, why = buy_decision(features, debug=debug)
        reasons.extend(why)
        if buy:
            guard_decision = "BUY"
    elif position_state == "long":
        sell, why = sell_decision(features, debug=debug)
        reasons.extend(why)
        if sell:
            guard_decision = "SELL"

    decision = guard_decision or decision_from_score

    if debug:
        print(f"[ARBITER] Score={score:+.2f} {decision}")
        print("  Contributions:")
        top = sorted(contribs, key=lambda x: abs(x[4]), reverse=True)[:5]
        for name, val, weight, reg_used, contrib in top:
            tag = reg_used if reg_used != "unused" else "unused"
            print(f"    {name} x{weight:.2f} ({tag}) = {contrib:+.2f}")

    if return_score:
        return decision, reasons, score, features
    return decision, reasons, features
