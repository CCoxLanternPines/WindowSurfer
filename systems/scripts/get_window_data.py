def get_wave_window_data_df(df, window: str, candle_offset: int = 0) -> dict | None:
    """
    Return window-relative wave structure data for current price.

    Returns:
        dict with:
            - window: str (e.g. "3d")
            - floor: float
            - ceiling: float
            - range: float
            - price: float
            - position_in_window: float (0 = floor, 1 = ceiling)
    """
    from systems.utils.time import parse_cutoff

    if df is None or df.empty:
        return None

    try:
        curr_close = float(df.iloc[len(df) - candle_offset - 1]["close"])
        past_close = float(df.iloc[len(df) - candle_offset - 25]["close"])
        # Normalize to % change: (current - past) / past
        trend_direction_delta_24h = ((curr_close - past_close) / past_close) * 100 if past_close else 0.0
    except (IndexError, KeyError, ZeroDivisionError):
        trend_direction_delta_24h = 0.0

    duration = parse_cutoff(window)
    num_candles = int(duration.total_seconds() // 3600)

    start_idx = max(0, len(df) - candle_offset - num_candles)
    end_idx = len(df) - candle_offset if candle_offset != 0 else None
    window_df = df.iloc[start_idx:end_idx]

    if window_df.empty:
        return None

    try:
        last_candle = df.iloc[-1 - candle_offset]
        price = float(last_candle["close"])
    except IndexError:
        return None

    floor = float(window_df["low"].min())
    ceiling = float(window_df["high"].max())
    range_val = ceiling - floor
    position = (price - floor) / range_val if range_val != 0 else 0.5

    return {
        "window": window,
        "floor": round(floor, 6),
        "ceiling": round(ceiling, 6),
        "range": round(range_val, 6),
        "price": round(price, 6),
        "position_in_window": round(position, 4),
        "trend_direction_delta_24h": trend_direction_delta_24h
    }
