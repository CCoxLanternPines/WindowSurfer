def should_buy_whale(candle, window_data, tick, cooldowns) -> bool:
    tunnel_pos = window_data.get("tunnel_position")
    if cooldowns["whale_catch"] <= 0 and tunnel_pos is not None and tunnel_pos < 0.1:
        return True
    return False
