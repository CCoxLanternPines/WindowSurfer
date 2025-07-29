from systems.decision_logic.fish_catch import should_sell_fish
from systems.decision_logic.whale_catch import should_sell_whale
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

    addlog(
        f"[EVAL] Evaluating Sell for {tag} 🐟🐋🔪",
        verbose_int=2,
        verbose_state=verbose,
    )


    # Separate notes by strategy so each strategy's logic only considers its own trades
    fish_notes = [n for n in notes if n.get("strategy") == "fish_catch"]
    whale_notes = [n for n in notes if n.get("strategy") == "whale_catch"]
    knife_notes = [n for n in notes if n.get("strategy") == "knife_catch"]

    knife_group_sell: list[dict] = []
    if len(knife_notes) >= 3:
        current_price = candle["close"]
        current_roi = lambda n: (current_price * n["entry_amount"] - n["entry_usdt"]) / n["entry_usdt"]
        highest_knife_roi = max(current_roi(n) for n in knife_notes)
        margin = SETTINGS.get("knife_group_roi_margin", 0.0)
        current_roi_trigger = (
            current_price * sum(n["entry_amount"] for n in knife_notes)
            - sum(n["entry_usdt"] for n in knife_notes)
        ) / sum(n["entry_usdt"] for n in knife_notes)
        addlog(
            f"[KNIFE GROUP] {len(knife_notes)} open | Highest ROI: {highest_knife_roi:.2%} | Trigger ROI: {current_roi_trigger:.2%}",
            verbose_int=2,
            verbose_state=verbose,
        )
        if current_roi_trigger >= highest_knife_roi + margin:
            knife_group_sell = knife_notes

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

    sell_list.extend(knife_group_sell)

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
