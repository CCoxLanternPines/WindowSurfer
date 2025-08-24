"""Utilities for formatting top-of-hour live trading reports."""

from datetime import datetime
from typing import Dict, Tuple


def format_top_of_hour_report(
    symbol: str,
    ts: datetime,
    usd_balance: float,
    coin_balance_usd: float,
    coin_symbol: str,
    total_liquid_value: float,
    triggered_strategies: Dict[str, bool],
    note_counts: Dict[str, Tuple[int, int]],
) -> str:
    """Return a formatted one line summary for the current hour."""

    def strat_emoji(name: str) -> str:
        if triggered_strategies.get(name, False):
            return {"Fish": "🐟", "Whale": "🐳", "Knife": "🔪"}.get(name, "❓")
        return "❌"

    fish = strat_emoji("Fish")
    whale = strat_emoji("Whale")
    knife = strat_emoji("Knife")
    trigger_str = f"🎣{fish}{whale}{knife}" if any(triggered_strategies.values()) else "❌❌❌"

    def fmt_notes(name: str) -> str:
        open_n, closed_n = note_counts.get(name, (0, 0))
        emoji = {"Fish": "🐟", "Whale": "🐳", "Knife": "🔪"}.get(name, name)
        return f"{emoji} {open_n}/{closed_n}"

    notes_str = " | ".join([fmt_notes("Fish"), fmt_notes("Whale"), fmt_notes("Knife")])
    hour_str = ts.strftime("%I:%M%p")

    return (
        f"[My Ledger] {hour_str} | {symbol} | 💰${total_liquid_value:.2f} | "
        f"💵${usd_balance:.2f} | 🪙${coin_balance_usd:.2f} | {trigger_str} | Notes: {notes_str}"
    )
