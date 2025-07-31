from systems.decision_logic.fish_catch import should_sell_notes as fish_should_sell_notes
from systems.decision_logic.whale_catch import should_sell_notes as whale_should_sell_notes
from systems.decision_logic.knife_catch import should_sell_notes as knife_should_sell_notes
from tqdm import tqdm
from systems.utils.logger import addlog
from systems.scripts.loader import load_settings

SETTINGS = load_settings()
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


    # Separate notes by strategy for routing
    fish_notes = [n for n in notes if n.get("strategy") == "fish_catch"]
    whale_notes = [n for n in notes if n.get("strategy") == "whale_catch"]
    knife_notes = [n for n in notes if n.get("strategy") == "knife_catch"]

    active_strategies = {n.get("strategy") for n in notes}

    # Provide window data and gain requirement to strategy functions via settings
    strategy_settings = {
        **SETTINGS,
        "window_data": window_data,
        "min_gain_pct": MIN_GAIN_PCT,
    }

    if "fish_catch" in active_strategies:
        sell_list += fish_should_sell_notes(fish_notes, candle, strategy_settings, verbose)

    if "whale_catch" in active_strategies:
        sell_list += whale_should_sell_notes(whale_notes, candle, strategy_settings, verbose)

    if "knife_catch" in active_strategies:
        sell_list += knife_should_sell_notes(knife_notes, candle, strategy_settings, verbose)

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
