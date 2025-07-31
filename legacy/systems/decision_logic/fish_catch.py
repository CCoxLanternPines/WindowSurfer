def should_buy_fish(candle, window_data, tick, cooldowns) -> bool:
    tunnel_pos = window_data.get("tunnel_position")
    if cooldowns["fish_catch"] <= 0 and tunnel_pos is not None and tunnel_pos < 0.6:
        return True
    return False


def should_sell_fish(candle, window_data, note) -> bool:
    tunnel_pos = window_data.get("tunnel_position")
    if tunnel_pos is not None and tunnel_pos > 0.9:
        return True
    return False


def should_sell_notes(
    notes: list,
    candle: dict,
    settings: dict,
    verbose: int = 0,
) -> list:
    """Evaluate a list of notes and return those that should be sold."""

    from systems.utils.logger import addlog

    min_gain_pct = settings.get("min_gain_pct", 0.05)
    window_data = settings.get("window_data", {})
    sell_list: list = []

    for note in notes:
        if note.get("strategy") != "fish_catch":
            continue

        entry_price = note.get("entry_price")
        if not entry_price:
            continue

        current_price = candle.get("close")
        gain_pct = (current_price - entry_price) / entry_price

        if gain_pct < min_gain_pct:
            addlog(
                f"[HOLD] {note['strategy']} | Gain {gain_pct:.2%} < Min Gain",
                verbose_int=2,
                verbose_state=verbose,
            )
            continue

        if should_sell_fish(candle, window_data, note):
            sell_list.append(note)

    return sell_list
