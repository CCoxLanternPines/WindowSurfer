from __future__ import annotations

"""Live engine mirroring the simulation strategy."""

import time
from datetime import datetime, timezone
from typing import Dict

from tqdm import tqdm

from systems.scripts.fetch_canles import fetch_candles
from systems.scripts.ledger import Ledger, save_ledger
from systems.scripts.evaluate_buy import evaluate_buy_signal
from systems.scripts.evaluate_sell import evaluate_sell_actions
from systems.utils.addlog import addlog
from systems.utils.config import load_settings


def _run_iteration(settings, runtime_states, *, dry: bool, verbose: int) -> None:
    for name, ledger_cfg in settings.get("ledger_settings", {}).items():
        tag = ledger_cfg.get("tag", "").upper()
        cfg = next(iter(ledger_cfg.get("window_settings", {}).values()))
        try:
            df = fetch_candles(tag)
        except FileNotFoundError:
            addlog(
                f"[WARN] Candle data missing for {tag}",
                verbose_int=1,
                verbose_state=verbose,
            )
            continue
        if df.empty:
            continue
        t = len(df) - 1
        ledger_obj = Ledger.load_ledger(tag=ledger_cfg["tag"])
        state = runtime_states.setdefault(
            name,
            {
                "capital": float(settings.get("simulation_capital", 0.0)),
                "buy_unlock_p": None,
                "verbose": verbose,
            },
        )

        ctx = {"ledger": ledger_obj}
        price = float(df.iloc[t]["close"])
        buy_res = evaluate_buy_signal(ctx, t, df, cfg, state)
        if buy_res:
            size_usd = buy_res["size_usd"]
            note_meta = buy_res["note"]
            amount = size_usd / price if price else 0.0
            note = {
                "id": f"{name}-{t}",
                "window": "default",
                "entry_idx": t,
                "entry_price": price,
                "entry_usdt": size_usd,
                "entry_amount": amount,
                **note_meta,
            }
            ledger_obj.open_note(note)
            state["capital"] -= size_usd
            state["buy_unlock_p"] = note_meta["unlock_p"]

        open_notes = ledger_obj.get_open_notes()
        sell_notes = evaluate_sell_actions(ctx, t, df, cfg, open_notes, state)
        sell_notes = sell_notes[: cfg.get("max_notes_sell_per_candle", 1)]
        for note in sell_notes:
            exit_usdt = note["entry_amount"] * price
            note["exit_idx"] = t
            note["exit_price"] = price
            note["exit_usdt"] = exit_usdt
            note["gain"] = exit_usdt - note["entry_usdt"]
            note["gain_pct"] = (
                note["gain"] / note["entry_usdt"] if note["entry_usdt"] else 0.0
            )
            ledger_obj.close_note(note)
            state["capital"] += exit_usdt

        save_ledger(ledger_cfg["tag"], ledger_obj)


def run_live(*, dry: bool = False, verbose: int = 0) -> None:
    settings = load_settings()
    runtime_states: Dict[str, Dict] = {}

    if dry:
        _run_iteration(settings, runtime_states, dry=dry, verbose=verbose)
        return

    while True:
        now = datetime.utcnow().replace(tzinfo=timezone.utc)
        elapsed = now.minute * 60 + now.second
        remaining = 3600 - elapsed
        with tqdm(
            total=3600,
            initial=elapsed,
            desc="⏳ Time to next hour",
            bar_format="{l_bar}{bar}| {percentage:3.0f}% {remaining}s",
            leave=True,
            dynamic_ncols=True,
        ) as pbar:
            for _ in range(remaining):
                time.sleep(1)
                pbar.update(1)
        addlog("[LIVE] Running top of hour", verbose_int=1, verbose_state=verbose)
        _run_iteration(settings, runtime_states, dry=dry, verbose=verbose)
