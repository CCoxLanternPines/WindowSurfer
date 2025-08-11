from __future__ import annotations

"""Utilities to perform a live-mode smoke test using paper execution."""

import json
import time
from typing import Dict, Any

from systems.scripts.ledger import Ledger
from systems.scripts.trade_apply import (
    paper_execute_buy,
    paper_execute_sell,
    apply_buy_result_to_ledger,
    apply_sell_result_to_ledger,
)
from systems.utils.addlog import addlog
from systems.utils.config import resolve_path


def _copy_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow copy of ``state`` with nested dicts copied."""
    out: Dict[str, Any] = {}
    for k, v in state.items():
        if isinstance(v, dict):
            out[k] = dict(v)
        else:
            out[k] = v
    return out


def run_smoke_test(
    *,
    ledger_name: str,
    ledger_cfg: Dict[str, Any],
    settings: Dict[str, Any],
    candle: Dict[str, Any],
    state: Dict[str, Any],
    save: bool,
    verbose: int,
) -> Dict[str, Any]:
    """Execute a paper buy then sell to exercise the trade pipeline."""

    tmp_state = _copy_state(state)
    limits = tmp_state.get("limits", {})
    min_sz = float(limits.get("min_note_size", 0.0))
    max_sz = float(limits.get("max_note_usdt", float("inf")))
    capital = float(tmp_state.get("capital", 0.0))
    size_usd = min(max(min_sz, min(capital, max_sz)), capital)
    if size_usd < min_sz:
        addlog("[SMOKE][SKIP] size < min", verbose_int=1, verbose_state=verbose)
        return {}

    price = float(candle.get("close", 0.0))
    ts = int(candle.get("timestamp", 0)) or None

    buy_res = paper_execute_buy(price, size_usd, timestamp=ts)
    ledger = Ledger()
    meta = {
        "window_name": "SMOKE",
        "window_size": 0,
    }
    note = apply_buy_result_to_ledger(
        ledger=ledger,
        window_name="SMOKE",
        t=0,
        meta=meta,
        result=buy_res,
        state=tmp_state,
    )
    addlog(
        f"[SMOKE][BUY] size=${size_usd:.2f}, amount={buy_res['filled_amount']:.8f}, price=${buy_res['avg_price']:.2f}, ts={buy_res['timestamp']}",
        verbose_int=1,
        verbose_state=verbose,
    )

    sell_price = max(buy_res["avg_price"], price)
    sell_res = paper_execute_sell(sell_price, buy_res["filled_amount"], timestamp=ts)
    apply_sell_result_to_ledger(
        ledger=ledger,
        note=note,
        t=1,
        result=sell_res,
        state=tmp_state,
    )
    buy_cost = buy_res["filled_amount"] * buy_res["avg_price"]
    sell_proceeds = sell_res["filled_amount"] * sell_res["avg_price"]
    gain = sell_proceeds - buy_cost
    roi = (gain / buy_cost * 100.0) if buy_cost > 0 else 0.0
    addlog(
        f"[SMOKE][SELL] amount={sell_res['filled_amount']:.8f}, price=${sell_res['avg_price']:.2f}, gain=${gain:.2f}, roi={roi:.2f}%",
        verbose_int=1,
        verbose_state=verbose,
    )

    summary = ledger.get_account_summary(sell_res["avg_price"])
    addlog(f"[SMOKE][SUMMARY] {summary}", verbose_int=1, verbose_state=verbose)

    path = None
    if save:
        root = resolve_path("")
        out_dir = root / "data" / "tmp"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts_save = int(time.time())
        out_path = out_dir / f"ledger-smoke-{ledger_name}-{ts_save}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "open_notes": ledger.get_open_notes(),
                    "closed_notes": ledger.get_closed_notes(),
                },
                f,
                indent=2,
            )
        path = str(out_path)

    return {"path": path, "summary": summary}
