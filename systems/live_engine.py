from __future__ import annotations

"""Live engine mirroring the simulation strategy."""

import time
from datetime import datetime, timezone
from typing import Dict

from tqdm import tqdm

from systems.scripts.fetch_canles import fetch_candles
from systems.scripts.ledger import Ledger, save_ledger
from systems.scripts.evaluate_buy import evaluate_buy
from systems.scripts.evaluate_sell import evaluate_sell
from systems.scripts.runtime_state import build_runtime_state
from systems.scripts.trade_apply import apply_buy_result_to_ledger, apply_sell_result_to_ledger
from systems.scripts.execution_handler import execute_buy, execute_sell
from systems.utils.addlog import addlog
from systems.utils.config import load_settings


def _run_iteration(settings, runtime_states, *, dry: bool, verbose: int) -> None:
    for name, ledger_cfg in settings.get("ledger_settings", {}).items():
        tag = ledger_cfg.get("tag", "").upper()
        window_settings = ledger_cfg.get("window_settings", {})
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
        prev = runtime_states.get(name, {"verbose": verbose})
        state = build_runtime_state(
            settings,
            ledger_cfg,
            mode="live",
            prev=prev,
        )
        runtime_states[name] = state

        price = float(df.iloc[t]["close"])
        for window_name, wcfg in window_settings.items():
            ctx = {"ledger": ledger_obj}
            buy_res = evaluate_buy(
                ctx,
                t,
                df,
                window_name=window_name,
                cfg=wcfg,
                runtime_state=state,
            )
            if buy_res:
                result = execute_buy(
                    None,
                    symbol=ledger_cfg["tag"],
                    price=price,
                    amount_usd=buy_res["size_usd"],
                    ledger_name=ledger_cfg["tag"],
                    wallet_code=ledger_cfg.get("wallet_code", ""),
                    verbose=state.get("verbose", 0),
                )
                if result:
                    apply_buy_result_to_ledger(
                        ledger=ledger_obj,
                        window_name=window_name,
                        t=t,
                        meta=buy_res,
                        result=result,
                        state=state,
                    )

            open_notes = ledger_obj.get_open_notes()
            sell_notes = evaluate_sell(
                ctx,
                t,
                df,
                window_name=window_name,
                cfg=wcfg,
                open_notes=open_notes,
                runtime_state=state,
            )
            for note in sell_notes:
                result = execute_sell(
                    None,
                    symbol=ledger_cfg["tag"],
                    coin_amount=note.get("entry_amount", 0.0),
                    price=price,
                    ledger_name=ledger_cfg["tag"],
                    verbose=state.get("verbose", 0),
                )
                if result:
                    apply_sell_result_to_ledger(
                        ledger=ledger_obj,
                        note=note,
                        t=t,
                        result=result,
                        state=state,
                    )

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
