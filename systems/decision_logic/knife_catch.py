from systems.utils.logger import addlog

def should_buy_knife(candle, window_data, tick, cooldowns, last_window_position=None) -> bool:
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
    to_sell = []

    # Loop through knife notes for detailed ROI and price comparison
    for i, note in enumerate(notes):
        entry_price = note["entry_usdt"] / note["entry_amount"]
        roi = (candle["close"] * note["entry_amount"] - note["entry_usdt"]) / note["entry_usdt"]
        current_price = candle["close"]

        addlog(
            f"[KNIFE SELL DEBUG] Note {i} | ROI: {roi:.2%} | Entry @ ${entry_price:.5f} | Now @ ${current_price:.5f} | Target Delta: {settings.get('knife_group_roi_margin', 0.30):.2%}",
            verbose_int=0,
            verbose_state=verbose
        )
        
        
    if len(notes) >= 3:
        current_price = candle["close"]
        margin = settings.get("knife_group_roi_margin", 0.30)

        current_roi = lambda n: (current_price * n["entry_amount"] - n["entry_usdt"]) / n["entry_usdt"]
        highest_knife_roi = max(current_roi(n) for n in notes)
        trigger_roi = (
            current_price * sum(n["entry_amount"] for n in notes)
            - sum(n["entry_usdt"] for n in notes)
        ) / sum(n["entry_usdt"] for n in notes)

        addlog(
            f"[KNIFE GROUP] {len(notes)} open | Highest ROI: {highest_knife_roi:.2%} | Trigger ROI: {trigger_roi:.2%} | Margin: {margin:.2%}",
            verbose_int=2,
            verbose_state=verbose,
        )

        for i, n in enumerate(notes):
            roi = current_roi(n)
            addlog(
                f"[KNIFE DEBUG] Note {i} ROI: {roi:.2%} (Entry @ {n['entry_usdt']:.4f})",
                verbose_int=3,
                verbose_state=verbose,
            )

        if trigger_roi >= highest_knife_roi + margin:
            to_sell = notes

    return to_sell
