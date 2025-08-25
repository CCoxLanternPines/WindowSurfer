from __future__ import annotations
from typing import Any, Dict, List
from systems.utils.addlog import addlog

def evaluate_sell(
    ctx: Dict[str, Any],
    t: int,
    series,
    *,
    cfg: Dict[str, Any],
    open_notes: List[Dict[str, Any]],
    runtime_state: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Source-of-truth sell evaluation.
    Returns a list of normalized SELL dicts.
    """

    candle = series.iloc[t]
    price = float(candle["close"])
    sells = []

    for note in open_notes:
        target = note.get("sell_price")
        if target and price >= target:
            qty = note.get("units", 0.0)
            usd_value = qty * price
            addlog(
                f"[SELL] idx={t} hit target {target:.5f} now={price:.5f} usd={usd_value:.2f}",
                verbose_int=1,
                verbose_state=runtime_state.get("verbose", 0),
            )
            sells.append({
                "side": "SELL",
                "entry_price": note.get("entry_price"),
                "size_usd": usd_value,
                "units": qty,
                "created_idx": note.get("created_idx", t),
                "created_ts": int(candle.get("timestamp", 0)),
                "sell_price": price,
            })

    return sells
