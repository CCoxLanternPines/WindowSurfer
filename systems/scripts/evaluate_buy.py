# Pressure Bot Buy Knobs
DROP_SCALE = 0.005      # how deep below anchor before pressure = 1.0
AGGRESSIVENESS = 1.0    # scales buy size
SLOPE_MIN = 0.0         # slope must be >= this to allow buys
MIN_NOTE_SIZE = 10.0    # minimum note size in USD
WINDOW_SIZE = 48        # number of candles to calculate anchor/slope


def evaluate_buy(candle, state):
    price = float(candle["close"])
    closes = state.get("recent_closes", [])
    closes.append(price)
    if len(closes) > WINDOW_SIZE:
        closes.pop(0)
    state["recent_closes"] = closes

    # Update anchor
    anchor = max(closes) if closes else price
    state["anchor_price"] = anchor

    # Update slope
    if len(closes) >= 2:
        slope = (closes[-1] - closes[0]) / len(closes)
    else:
        slope = 0.0
    state["slope_direction_avg"] = slope

    # Update pressure
    drop = max(0.0, (anchor - price) / anchor)
    pressure = drop / DROP_SCALE
    state["pressure"] = pressure

    # Check buy conditions
    if slope >= SLOPE_MIN and pressure >= 1.0:
        size_usd = state.get("capital", 0) * AGGRESSIVENESS
        if size_usd >= MIN_NOTE_SIZE:
            note = {
                "entry_price": price,
                "entry_usdt": size_usd,
                "size_usd": size_usd,
                "reason": "PRESSURE_BUY",
            }
            state["pressure"] = 0.0  # reset pressure
            return note

    return None
