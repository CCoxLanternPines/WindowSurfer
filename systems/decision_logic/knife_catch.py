def should_buy_knife(candle, window_data, tick, cooldowns, last_window_position=None) -> bool:
    tunnel_pos = window_data.get("tunnel_position")
    window_pos = window_data.get("window_position")
    
    if cooldowns["knife_catch"] <= 0:
        if tunnel_pos is not None and abs(tunnel_pos) < 0.01:
            if last_window_position is None or window_pos < last_window_position:
                return True
    return False


def should_sell_knife(candle, window_data, note, verbose: int = 0) -> bool:
    tunnel_pos = window_data.get("tunnel_position")
    window_pos = window_data.get("window_position")
    saved_window_pos = note.get("window_position_at_entry")
    
    if tunnel_pos is not None and tunnel_pos > 0.90:
        if saved_window_pos is not None and window_pos >= saved_window_pos:
            from systems.utils.logger import addlog
            addlog("[KNIFE] ✅ SELL triggered", verbose_int=1, verbose_state=verbose)
            return True
    return False


def should_sell_notes(
    notes: list,
    candle: dict,
    settings: dict,
    verbose: int = 0,
) -> list:
    """Return knife_catch notes that should be sold this tick."""

    from systems.utils.logger import addlog

    knife_notes = [n for n in notes if n.get("strategy") == "knife_catch"]
    to_sell: list = []

    if len(knife_notes) >= 3:
        current_price = candle["close"]
        current_roi = lambda n: (
            current_price * n["entry_amount"] - n["entry_usdt"]
        ) / n["entry_usdt"]

        highest_roi = max(current_roi(n) for n in knife_notes)
        margin = settings.get("knife_group_roi_margin", 0.0)
        trigger_roi = (
            current_price * sum(n["entry_amount"] for n in knife_notes)
            - sum(n["entry_usdt"] for n in knife_notes)
        ) / sum(n["entry_usdt"] for n in knife_notes)

        addlog(
            f"[KNIFE GROUP] {len(knife_notes)} open | Highest ROI: {highest_roi:.2%} | Trigger ROI: {trigger_roi:.2%}",
            verbose_int=2,
            verbose_state=verbose,
        )

        if trigger_roi >= highest_roi + margin:
            to_sell = knife_notes

    return to_sell
