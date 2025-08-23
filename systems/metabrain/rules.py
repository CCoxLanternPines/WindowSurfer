"""Guardrail rule implementations for MetaBrain decisions.

These simple heuristics act as safety checks before weighted scoring. The
behavior remains unchanged from prior iterations.
"""


from __future__ import annotations

from typing import Dict, List, Tuple


def buy_decision(features: Dict[str, float], debug: bool = False) -> Tuple[bool, List[str]]:
    """Return (decision, reasons) for BUY guardrail."""
    reasons = []
    val1 = features.get("exh_edge_accuracy", 0)
    thresh1 = 60
    cond1 = val1 > thresh1
    if debug:
        reasons.append(
            f"exh_edge_accuracy - {val1:.2f} > {thresh1} {'PASS' if cond1 else 'FAIL'}"
        )

    val2 = features.get("flip_extrema_pct", 0)
    thresh2 = 50
    cond2 = val2 > thresh2
    if debug:
        reasons.append(
            f"flip_extrema_pct - {val2:.2f} > {thresh2} {'PASS' if cond2 else 'FAIL'}"
        )

    decision = cond1 and cond2
    return decision, reasons


def sell_decision(features: Dict[str, float], debug: bool = False) -> Tuple[bool, List[str]]:
    """Return (decision, reasons) for SELL guardrail."""
    reasons = []
    val1 = features.get("divergence_to_top", 0)
    thresh1 = 70
    cond1 = val1 > thresh1
    if debug:
        reasons.append(
            f"divergence_to_top - {val1:.2f} > {thresh1} {'PASS' if cond1 else 'FAIL'}"
        )

    val2 = features.get("peak_continuation", 100)
    thresh2 = 30
    cond2 = val2 < thresh2
    if debug:
        reasons.append(
            f"peak_continuation - {val2:.2f} < {thresh2} {'PASS' if cond2 else 'FAIL'}"
        )

    decision = cond1 and cond2
    return decision, reasons
