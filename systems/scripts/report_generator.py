"""Unified report aggregation for strategy performance."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict

from systems.utils.addlog import addlog


def compute_strategy_report(ledger, price_now_by_tag: Dict[str, float]) -> Dict[str, Dict]:
    """Aggregate per-strategy stats from ``ledger``.

    Parameters
    ----------
    ledger: Ledger
        Ledger instance containing notes and trades.
    price_now_by_tag: Dict[str, float]
        Mapping of strategy name to current price. If a strategy is missing
        from the mapping, the first value in the mapping is used as a
        fallback (assuming single-asset environments).
    """
    default_price = next(iter(price_now_by_tag.values()), 0.0)
    stats = defaultdict(
        lambda: {
            "buys": 0,
            "sells": 0,
            "gross_invested": 0.0,
            "realized_cost": 0.0,
            "realized_proceeds": 0.0,
            "realized_pnl": 0.0,
            "realized_roi": 0.0,
            "avg_trade_roi": 0.0,
            "open_value_now": 0.0,
            "window_total_at_liq": 0.0,
            "window_size": None,
            "_roi_accum": 0.0,
            "_realized_trades": 0,
        }
    )

    # Open notes ------------------------------------------------------------
    for note in ledger.get_open_notes():
        strat = note.get("strategy")
        if not strat:
            note_id = note.get("id", "?")
            strat = note_id.split("-", 1)[0] if "-" in note_id else "unknown"
            note["strategy"] = strat
            addlog(f"[REPORT][FIXUP] inferred strategy {strat} for note {note_id}")
        st = stats[strat]
        st["buys"] += 1
        entry_usd = note.get("entry_usdt", 0.0)
        st["gross_invested"] += entry_usd
        qty = note.get("entry_amount", 0.0)
        price_now = price_now_by_tag.get(strat, default_price)
        st["open_value_now"] += qty * price_now
        if st["window_size"] is None and note.get("window_size") is not None:
            st["window_size"] = note.get("window_size")

    # Closed notes ----------------------------------------------------------
    for note in ledger.get_closed_notes():
        strat = note.get("strategy")
        if not strat:
            note_id = note.get("id", "?")
            strat = note_id.split("-", 1)[0] if "-" in note_id else "unknown"
            note["strategy"] = strat
            addlog(f"[REPORT][FIXUP] inferred strategy {strat} for note {note_id}")
        st = stats[strat]
        st["buys"] += 1
        st["sells"] += 1
        entry_usd = note.get("entry_usdt")
        if entry_usd is None:
            entry_usd = note.get("entry_amount", 0.0) * note.get("entry_price", 0.0)
        exit_usd = note.get("exit_usdt")
        if exit_usd is None:
            exit_usd = note.get("entry_amount", 0.0) * note.get("exit_price", 0.0)
        st["gross_invested"] += entry_usd
        st["realized_cost"] += entry_usd
        st["realized_proceeds"] += exit_usd
        pnl = exit_usd - entry_usd
        st["realized_pnl"] += pnl
        if entry_usd > 0:
            st["_roi_accum"] += pnl / entry_usd
        st["_realized_trades"] += 1
        if st["window_size"] is None and note.get("window_size") is not None:
            st["window_size"] = note.get("window_size")

    # Finalize --------------------------------------------------------------
    final_stats: Dict[str, Dict] = {}
    for strat, st in stats.items():
        if st["realized_cost"] > 0:
            st["realized_roi"] = st["realized_pnl"] / st["realized_cost"]
        if st["_realized_trades"] > 0:
            st["avg_trade_roi"] = st["_roi_accum"] / st["_realized_trades"]
        st["window_total_at_liq"] = st["realized_proceeds"] + st["open_value_now"]
        st.pop("_roi_accum", None)
        st.pop("_realized_trades", None)
        final_stats[strat] = st
    return final_stats
