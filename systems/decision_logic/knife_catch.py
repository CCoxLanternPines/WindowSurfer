def should_buy_knife(candle, window_data, tick, cooldowns, last_window_position=None) -> bool:
    tunnel_pos = window_data.get("tunnel_position")
    window_pos = window_data.get("window_position")
    
    if cooldowns["knife_catch"] <= 0:
        if tunnel_pos is not None and abs(tunnel_pos) < 0.01:
            if last_window_position is None or window_pos < last_window_position:
                return True
    return False


def should_sell_knife(candle, window_data, note) -> bool:
    tunnel_pos = window_data.get("tunnel_position")
    window_pos = window_data.get("window_position")
    saved_window_pos = note.get("window_position_at_entry")

    if tunnel_pos is not None and tunnel_pos > 0.9:
        if saved_window_pos is not None and window_pos >= saved_window_pos:
            return True
    return False
