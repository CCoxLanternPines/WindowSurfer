def evaluate_sell(candle, notes, state):
    price = float(candle["close"])
    anchor = state.get("anchor_price", price)
    sells = []

    # Flat sell
    flat_threshold = state.get("flat_threshold", 0.03)
    trigger = anchor * (1.0 - flat_threshold)
    if price <= trigger and notes:
        notes_sorted = sorted(notes, key=lambda n: (price - n["entry_price"]) / n["entry_price"], reverse=True)
        half = len(notes_sorted) // 2
        for n in notes_sorted[:half]:
            sells.append({"note_id": n["id"], "reason": "FLAT_SELL"})

    # Full sell
    if state.get("force_sell", False):
        for n in notes:
            sells.append({"note_id": n["id"], "reason": "SELL_FULL"})

    return sells

