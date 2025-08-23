def buy_decision(features, debug=False):
    reasons = []
    ok1 = features.get("exh_edge_accuracy", 0) > 60
    ok2 = features.get("flip_extrema_pct", 0) > 50
    if debug:
        reasons.append(
            f"exh_edge_accuracy={features.get('exh_edge_accuracy')} {'OK' if ok1 else 'FAIL'}"
        )
        reasons.append(
            f"flip_extrema_pct={features.get('flip_extrema_pct')} {'OK' if ok2 else 'FAIL'}"
        )
    return (ok1 and ok2), reasons


def sell_decision(features, debug=False):
    reasons = []
    ok1 = features.get("divergence_to_top", 0) > 70
    ok2 = features.get("peak_continuation", 100) < 30
    if debug:
        reasons.append(
            f"divergence_to_top={features.get('divergence_to_top')} {'OK' if ok1 else 'FAIL'}"
        )
        reasons.append(
            f"peak_continuation={features.get('peak_continuation')} {'OK' if ok2 else 'FAIL'}"
        )
    return (ok1 and ok2), reasons

