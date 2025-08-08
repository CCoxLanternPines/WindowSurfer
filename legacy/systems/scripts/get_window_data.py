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
        from systems.utils.time import parse_cutoff
        window_duration = parse_cutoff(window)
        num_candles = int(window_duration.total_seconds() // 3600)

        curr_close = float(df.iloc[len(df) - candle_offset - 1]["close"])
        past_close = float(df.iloc[len(df) - candle_offset - num_candles]["close"])
        # Normalize to % change: (current - past) / past
        trend_direction_delta_window = ((curr_close - past_close) / past_close) * 100 if past_close else 0.0
    except (IndexError, KeyError, ZeroDivisionError):
        trend_direction_delta_window = 0.0


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
        "trend_direction_delta_window": trend_direction_delta_window
    }


def get_window_data(*, wave: dict, price: float) -> dict:
    """Return averaged tunnel metrics for logging purposes.

    Parameters
    ----------
    wave:
        Dictionary containing at least ``floor`` and ``ceiling`` keys.
    price:
        Current asset price.

    Returns
    -------
    dict
        Mapping with ``current_tunnel_position_avg``, ``loudness_avg``,
        ``slope_direction_avg`` and ``highest_spike_avg``.
    """

    floor = wave.get("floor", 0.0)
    ceiling = wave.get("ceiling", 0.0)
    range_val = ceiling - floor

    position = ((price - floor) / range_val * 2 - 1) if range_val else 0.0
    loudness = (range_val / price) if price else 0.0
    slope = wave.get("trend_direction_delta_window", 0.0)
    highest_spike = (
        max(abs(price - floor), abs(ceiling - price)) / price if price else 0.0
    )

    return {
        "current_tunnel_position_avg": round(position, 2),
        "loudness_avg": round(loudness, 2),
        "slope_direction_avg": round(slope, 2),
        "highest_spike_avg": round(highest_spike, 2),
    }
