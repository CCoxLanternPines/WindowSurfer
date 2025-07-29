from systems.decision_logic.fish_catch import should_sell_fish
from systems.decision_logic.whale_catch import should_sell_whale
from systems.decision_logic.knife_catch import should_sell_knife
from tqdm import tqdm
from systems.utils.logger import addlog
MIN_GAIN_PCT = 0.05  # Require at least +5% ROI to sell

def evaluate_sell_df(
    candle: dict,
    window_data: dict,
    tick: int,
    notes: list[dict],
    tag: str,
    verbose: int = 0
) -> list[dict]:
    """Given current market state and open notes, returns list of notes to be sold."""
    sell_list = []

    addlog(
        f"[EVAL] Evaluating Sell for {tag} 🐟🐋🔪",
        verbose_int=2,
        verbose_state=verbose,
    )


    # Separate notes by strategy so each strategy's logic only considers its own trades
    fish_notes = [n for n in notes if n.get("strategy") == "fish_catch"]
    whale_notes = [n for n in notes if n.get("strategy") == "whale_catch"]
    knife_notes = [n for n in notes if n.get("strategy") == "knife_catch"]

    # Placeholder for future knife group exit logic
    # if knife_group_should_exit(knife_notes, candle):
    #     for n in knife_notes:
    #         sell_list.append(n)

    for note in fish_notes:
        entry_price = note.get("entry_price")
        if not entry_price:
            continue

        current_price = candle.get("close")
        gain_pct = (current_price - entry_price) / entry_price

        if gain_pct < MIN_GAIN_PCT:
            addlog(
                f"[HOLD] {note['strategy']} | Tick {tick} | Gain {gain_pct:.2%} < Min Gain",
                verbose_int=2,
                verbose_state=verbose,
            )
            continue

        if should_sell_fish(candle, window_data, note):
            sell_list.append(note)

    for note in whale_notes:
        entry_price = note.get("entry_price")
        if not entry_price:
            continue

        current_price = candle.get("close")
        gain_pct = (current_price - entry_price) / entry_price

        if gain_pct < MIN_GAIN_PCT:
            addlog(
                f"[HOLD] {note['strategy']} | Tick {tick} | Gain {gain_pct:.2%} < Min Gain",
                verbose_int=2,
                verbose_state=verbose,
            )
            continue

        if should_sell_whale(candle, window_data, note):
            sell_list.append(note)

    for note in knife_notes:
        entry_price = note.get("entry_price")
        if not entry_price:
            continue

        current_price = candle.get("close")
        gain_pct = (current_price - entry_price) / entry_price

        if gain_pct < MIN_GAIN_PCT:
            addlog(
                f"[HOLD] {note['strategy']} | Tick {tick} | Gain {gain_pct:.2%} < Min Gain",
                verbose_int=2,
                verbose_state=verbose,
            )
            continue

        if should_sell_knife(candle, window_data, note, verbose=verbose):
            sell_list.append(note)

    addlog(
        f"[EVAL] Active Notes: {len(notes)}",
        verbose_int=2,
        verbose_state=verbose,
    )

    sell_list.sort(
        key=lambda note: (
            note.get("entry_amount", 0) * candle["close"]
            - note.get("entry_usdt", 0)
        ),
        reverse=True,
    )

    for note in sell_list:
        projected_gain = note["entry_amount"] * candle["close"] - note["entry_usdt"]
        addlog(
            f"[PRIORITY SELL] Strategy: {note['strategy']} | Est Gain: ${projected_gain:.2f}",
            verbose_int=2,
            verbose_state=verbose,
        )

    return sell_list
