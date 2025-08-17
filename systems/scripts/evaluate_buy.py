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

    # Update pressure (cumulative)
    drop = (anchor - price) / anchor if anchor else 0.0
    state["pressure"] = max(0.0, state.get("pressure", 0.0) + drop)

    # Check buy conditions
    trigger = anchor * (1.0 - state["pressure"] * DROP_SCALE)
    if slope >= SLOPE_MIN and price <= trigger:
        note = {
            "entry_price": price,
            "entry_usdt": state.get("capital", 0) * AGGRESSIVENESS,
            "reason": "PRESSURE_BUY",
        }
        state["pressure"] = 0.0  # reset after buy
        return note

    return None
