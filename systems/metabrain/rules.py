def buy_decision(features: dict) -> bool:
    return (
        (features.get("exh_edge_accuracy", 0) > 60)
        and (features.get("flip_extrema_pct", 0) > 50)
    )


def sell_decision(features: dict) -> bool:
    return (
        (features.get("divergence_to_top", 0) > 70)
        and (features.get("peak_continuation", 100) < 30)
    )

