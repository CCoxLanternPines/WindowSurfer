# Pressure Bot Sell Knobs
FLAT_THRESHOLD = 0.03   # 3% drawdown from anchor triggers flat-sell


def evaluate_sell(candle, notes, state):
    price = float(candle["close"])
    anchor = state.get("anchor_price", price)

    sells = []

    # Flat sell: sell half the notes if price drops below threshold
    trigger = anchor * (1.0 - FLAT_THRESHOLD)
    if price <= trigger and notes:
        # Sort notes by ROI (highest first)
        notes_sorted = sorted(
            notes,
            key=lambda n: (price - n["entry_price"]) / n["entry_price"],
            reverse=True,
        )
        half = len(notes_sorted) // 2
        for n in notes_sorted[:half]:
            sells.append({"note_id": n["id"], "reason": "FLAT_SELL"})

    # Full sell: external trigger
    if state.get("force_sell", False):
        for n in notes:
            sells.append({"note_id": n["id"], "reason": "SELL_FULL"})

    return sells
