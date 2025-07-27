from systems.decision_logic.fish_catch import should_sell_fish
from systems.decision_logic.whale_catch import should_sell_whale
from systems.decision_logic.knife_catch import should_sell_knife

MIN_GAIN_PCT = 0.05  # Require at least +5% ROI to sell

def evaluate_sell_df(
    candle: dict,
    window_data: dict,
    tick: int,
    notes: list[dict],
    verbose: int = 0
) -> list[dict]:
    """Given current market state and open notes, returns list of notes to be sold."""
    sell_list = []

    for note in notes:
        entry_price = note.get("entry_price")
        if not entry_price:
            continue

        current_price = candle.get("close")
        gain_pct = (current_price - entry_price) / entry_price

        if gain_pct < MIN_GAIN_PCT:
            if verbose >= 2:
                print(f"[HOLD] {note['strategy']} | Tick {tick} | Gain {gain_pct:.2%} < Min Gain")
            continue

        strategy = note.get("strategy")

        if strategy == "fish_catch" and should_sell_fish(candle, window_data, note):
            sell_list.append(note)
        elif strategy == "whale_catch" and should_sell_whale(candle, window_data, note):
            sell_list.append(note)
        elif strategy == "knife_catch" and should_sell_knife(candle, window_data, note, verbose=verbose):
            sell_list.append(note)

    return sell_list
