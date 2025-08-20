from __future__ import annotations

from typing import Any, Dict, Optional


def paper_execute_buy(price: float, amount_usd: float, *, timestamp: Optional[int] = None) -> Dict[str, float | int | None]:
    filled = amount_usd / price if price else 0.0
    return {"filled_amount": filled, "avg_price": price, "timestamp": timestamp}


def paper_execute_sell(price: float, coin_amount: float, *, timestamp: Optional[int] = None) -> Dict[str, float | int | None]:
    return {"filled_amount": coin_amount, "avg_price": price, "timestamp": timestamp}


def apply_buy(
    *,
    ledger,
    window_name: str,
    t: int,
    meta: Dict[str, Any],
    result: Dict[str, Any],
    state: Dict[str, Any],
) -> Dict[str, Any]:
    note = {
        "id": f"{window_name}-{t}",
        "entry_idx": t,
        "entry_price": result.get("avg_price", 0.0),
        "entry_amount": result.get("filled_amount", 0.0),
        "entry_usdt": result.get("filled_amount", 0.0) * result.get("avg_price", 0.0),
        "window_name": meta.get("window_name"),
        "window_size": meta.get("window_size"),
        "p_buy": meta.get("p_buy"),
        "target_price": meta.get("target_price"),
        "target_roi": meta.get("target_roi"),
        "unlock_p": meta.get("unlock_p"),
    }
    if "created_idx" in meta:
        note["created_idx"] = meta["created_idx"]
    if result.get("timestamp") is not None:
        note["created_ts"] = result.get("timestamp")
    elif "created_ts" in meta:
        note["created_ts"] = meta["created_ts"]

    # Merge any additional metadata (e.g. note kind)
    for k, v in meta.items():
        if k not in note:
            note[k] = v

    ledger.open_note(note)
    cost = result.get("filled_amount", 0.0) * result.get("avg_price", 0.0)
    state["capital"] = state.get("capital", 0.0) - cost
    # Persist remaining capital on the ledger for live-mode parity
    try:
        ledger.set_metadata({"capital": state.get("capital", 0.0)})
    except AttributeError:
        pass
    return note


def apply_sell(
    *,
    ledger,
    note: Dict[str, Any],
    t: int | None,
    result: Dict[str, Any],
    state: Dict[str, Any],
) -> Dict[str, Any]:
    exit_price = result.get("avg_price", 0.0)
    exit_usdt = result.get("filled_amount", 0.0) * exit_price
    note["exit_price"] = exit_price
    note["exit_usdt"] = exit_usdt
    if t is not None:
        note["exit_idx"] = t
    if result.get("timestamp") is not None:
        note["exit_ts"] = result.get("timestamp")
    entry_usdt = note.get("entry_usdt", 0.0)
    note["gain"] = exit_usdt - entry_usdt
    note["gain_pct"] = (note["gain"] / entry_usdt) if entry_usdt else 0.0
    ledger.close_note(note)
    state["capital"] = state.get("capital", 0.0) + exit_usdt
    # Persist capital back to ledger after the sale
    try:
        ledger.set_metadata({"capital": state.get("capital", 0.0)})
    except AttributeError:
        pass
    return note
