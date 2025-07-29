from systems.utils.logger import addlog


def should_buy_knife(
    candle,
    window_data,
    tick,
    cooldowns,
    active_notes,
    last_window_position=None,
    settings=None,
    verbose: int = 0,
) -> bool:
    """Determine if a new knife note should be opened."""

    knife_limit = (
        settings.get("strategy_limits", {})
        .get("knife_catch", {})
        .get("max_open_notes", 99)
        if settings
        else 99
    )

    if len(active_notes) >= knife_limit:
        addlog(
            f"[KNIFE BLOCKED] Max knife notes reached: {knife_limit}",
            verbose_int=2,
            verbose_state=verbose,
        )
        return False

    tunnel_pos = window_data.get("tunnel_position")
    window_pos = window_data.get("window_position")

    if cooldowns["knife_catch"] <= 0:
        if tunnel_pos is not None and abs(tunnel_pos) < 0.01:
            if last_window_position is None or window_pos < last_window_position:
                return True
    return False


def should_sell_knife(candle, window_data, note, verbose: int = 0) -> bool:
    tunnel_pos = window_data.get("tunnel_position")
    window_pos = window_data.get("window_position")
    saved_window_pos = note.get("window_position_at_entry")
    
    if tunnel_pos is not None and tunnel_pos > 0.50:
        if saved_window_pos is not None and window_pos >= saved_window_pos:
            addlog("[KNIFE] ✅ SELL triggered", verbose_int=1, verbose_state=verbose)
            return True
    return False


def should_sell_notes(notes: list, candle: dict, settings: dict, verbose: int = 0) -> list:
    """Return all knife notes if any single note hits the ROI margin."""

    margin = settings.get("knife_group_roi_margin", 0.30)
    current_price = candle["close"]

    trigger_hit = False
    for i, note in enumerate(notes):
        roi = (current_price * note["entry_amount"] - note["entry_usdt"]) / note["entry_usdt"]
        addlog(
            f"[KNIFE SELL DEBUG] Note ROI: {roi:.2%} vs Margin: {margin:.2%}",
            verbose_int=2,
            verbose_state=verbose,
        )

        if roi < 0:
            addlog(
                f"[KNIFE SELL BLOCKED] Note {i} still negative ({roi:.2%}) \u2014 cancel group exit",
                verbose_int=1,
                verbose_state=verbose,
            )
            return []  # Block all

        if roi >= margin:
            trigger_hit = True

    if trigger_hit:
        addlog(
            "[KNIFE GROUP] \u2705 ROI target met & all notes positive \u2014 selling all",
            verbose_int=1,
            verbose_state=verbose,
        )
        return notes

    return []
