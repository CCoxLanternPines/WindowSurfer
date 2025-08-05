def get_trade_params(current_price, window_high, window_low, config, entry_price=None):
    """Compute position-based trade parameters and maturity ROI.

    Args:
        current_price (float): Current market price.
        window_high (float): Upper boundary of the window.
        window_low (float): Lower boundary of the window.
        config (dict): Strategy configuration containing multiplier settings.
        entry_price (float, optional): Entry price of a note for ROI calculation.

    Returns:
        dict: Dictionary with position percentage, buy and cooldown multipliers,
              and maturity ROI (None if entry_price is not provided).
    """
    window_range = window_high - window_low
    if window_range == 0:
        pos_pct = 0.0
    else:
        pos_pct = ((current_price - window_low) / window_range) * 2 - 1

    buy_scale = config.get("buy_multiplier_scale", 1.0)
    cooldown_scale = config.get("cooldown_multiplier_scale", 1.0)
    buy_multiplier = 1.0 + (abs(pos_pct) * (buy_scale - 1.0))
    cooldown_multiplier = 1.0 + (abs(pos_pct) * (cooldown_scale - 1.0))

    maturity_roi = None
    if entry_price is not None and window_range != 0:
        entry_pos_pct = ((entry_price - window_low) / window_range) * 2 - 1
        mirrored_pos_pct = -entry_pos_pct
        target_price = window_low + ((mirrored_pos_pct + 1) / 2) * window_range
        maturity_roi = (target_price - entry_price) / entry_price
        maturity_roi *= config.get("maturity_multiplier", 1.0)

    return {
        "pos_pct": pos_pct,
        "buy_multiplier": buy_multiplier,
        "cooldown_multiplier": cooldown_multiplier,
        "maturity_roi": maturity_roi,
    }
