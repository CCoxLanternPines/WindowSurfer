def update_pressure_state(state, price):
    anchor = state.get("anchor_price", price)
    anchor = max(anchor, price)
    state["anchor_price"] = anchor

    drop = (anchor - price) / anchor if anchor else 0.0
    state["pressure"] = max(0.0, state.get("pressure", 0.0) + drop)


def pressure_buy_signal(state, price):
    anchor = state.get("anchor_price", price)
    trigger = anchor * (1.0 - state["pressure"] * state.get("drop_scale", 0.005))
    return price <= trigger


def evaluate_buy(candle, state):
    price = float(candle["close"])
    update_pressure_state(state, price)
    if pressure_buy_signal(state, price):
        return {
            "entry_price": price,
            "entry_usdt": state["capital"],
            "reason": "PRESSURE_BUY"
        }
    return None

