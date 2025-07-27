from systems.decision_logic.fish_catch import should_sell_fish
from systems.decision_logic.whale_catch import should_sell_whale
from systems.decision_logic.knife_catch import should_sell_knife


def evaluate_sell_df(
    candle: dict,
    window_data: dict,
    tick: int,
    notes: list[dict],
    verbose: bool = False
) -> list[dict]:
    """Given current market state and open notes, returns list of notes to be sold."""
    sell_list = []
    for note in notes:
        strategy = note.get("strategy")
        if strategy == "fish_catch" and should_sell_fish(candle, window_data, note):
            sell_list.append(note)
        elif strategy == "whale_catch" and should_sell_whale(candle, window_data, note):
            sell_list.append(note)
        elif strategy == "knife_catch" and should_sell_knife(candle, window_data, note):
            sell_list.append(note)
    return sell_list
