def get_trade_params(current_price, window_high, window_low, config, entry_price=None):
    """Compute position metrics and multipliers for buy/sell decisions.

    Parameters
    ----------
    current_price:
        Current market price.
    window_high:
        Upper boundary of the price window.
    window_low:
        Lower boundary of the price window.
    config:
        Strategy configuration containing multiplier scales and optional
        ``dead_zone_pct`` and ``maturity_multiplier`` values.
    entry_price:
        Optional entry price used to calculate maturity ROI.

    Returns
    -------
    dict
        Dictionary containing ``pos_pct``, ``in_dead_zone``, buy multipliers,
        buy/sell cooldown multipliers, and ``maturity_roi`` (``None`` if
        ``entry_price`` is not provided).
    """

    window_range = window_high - window_low
    if window_range == 0:
        pos_pct = 0.0
    else:
        pos_pct = ((current_price - window_low) / window_range) * 2 - 1

    dead_zone_pct = config.get("dead_zone_pct", 0.0)
    dead_zone_half = dead_zone_pct / 2
    in_dead_zone = abs(pos_pct) <= dead_zone_half if dead_zone_pct > 0 else False

    buy_scale = config.get("buy_multiplier_scale", 1.0)
    buy_multiplier = 1.0 + (abs(pos_pct) * (buy_scale - 1.0))

    buy_cd_multiplier = 1.0 + (
        abs(pos_pct)
        * (config.get("buy_cooldown_multiplier_scale", 1.0) - 1.0)
    )
    sell_cd_multiplier = 1.0 + (
        abs(pos_pct)
        * (config.get("sell_cooldown_multiplier_scale", 1.0) - 1.0)
    )

    maturity_roi = None
    if entry_price is not None and window_range != 0:
        maturity_multiplier = config.get("maturity_multiplier", 1.0)
        mirrored_pos = -pos_pct
        target_price = window_low + ((mirrored_pos + 1) / 2) * window_range
        maturity_roi = ((target_price - entry_price) / entry_price) * maturity_multiplier

    return {
        "pos_pct": pos_pct,
        "in_dead_zone": in_dead_zone,
        "buy_multiplier": buy_multiplier,
        "buy_cooldown_multiplier": buy_cd_multiplier,
        "sell_cooldown_multiplier": sell_cd_multiplier,
        "maturity_roi": maturity_roi,
    }
