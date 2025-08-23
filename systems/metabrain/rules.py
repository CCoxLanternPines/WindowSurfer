def buy_decision(features, weights, debug=False):
    reasons = []
    checks = []

    if weights.get("exh_edge_accuracy", 1) > 0:
        val = features.get("exh_edge_accuracy")
        ok = (val or 0) > 60
        checks.append(ok)
        if debug:
            reasons.append(
                f"exh_edge_accuracy={val} {'OK' if ok else 'FAIL'}"
            )
    elif debug:
        reasons.append("exh_edge_accuracy ignored")

    if weights.get("flip_extrema_pct", 1) > 0:
        val = features.get("flip_extrema_pct")
        ok = (val or 0) > 50
        checks.append(ok)
        if debug:
            reasons.append(
                f"flip_extrema_pct={val} {'OK' if ok else 'FAIL'}"
            )
    elif debug:
        reasons.append("flip_extrema_pct ignored")

    return (all(checks) if checks else False), reasons


def sell_decision(features, weights, debug=False):
    reasons = []
    checks = []

    if weights.get("divergence_to_top", 1) > 0:
        val = features.get("divergence_to_top")
        ok = (val or 0) > 70
        checks.append(ok)
        if debug:
            reasons.append(
                f"divergence_to_top={val} {'OK' if ok else 'FAIL'}"
            )
    elif debug:
        reasons.append("divergence_to_top ignored")

    if weights.get("peak_continuation", 1) > 0:
        val = features.get("peak_continuation", 100)
        ok = (val or 100) < 30
        checks.append(ok)
        if debug:
            reasons.append(
                f"peak_continuation={val} {'OK' if ok else 'FAIL'}"
            )
    elif debug:
        reasons.append("peak_continuation ignored")

    return (all(checks) if checks else False), reasons

