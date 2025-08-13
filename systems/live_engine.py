from __future__ import annotations

"""Live engine mirroring the simulation strategy."""

import time
from datetime import datetime, timezone
from typing import Dict

from tqdm import tqdm

from systems.fetch import fetch_recent_coin
from systems.scripts.fetch_candles import load_coin_csv
from systems.scripts.ledger import Ledger, save_ledger
from systems.scripts.evaluate_buy import evaluate_buy
from systems.scripts.evaluate_sell import evaluate_sell
from systems.scripts.runtime_state import build_runtime_state
from systems.scripts.trade_apply import apply_sell_result_to_ledger
from systems.scripts.execution_handler import execute_sell, process_buy_signal
from systems.utils.addlog import addlog
from systems.utils.config import load_settings
from systems.utils.resolve_symbol import split_tag


def _run_iteration(settings, runtime_states, *, dry: bool, verbose: int) -> None:
    for name, ledger_cfg in settings.get("ledger_settings", {}).items():
        base, _ = split_tag(ledger_cfg["tag"])
        coin = base.upper()
        window_settings = ledger_cfg.get("window_settings", {})
        try:
            df = load_coin_csv(coin)
        except FileNotFoundError:
            addlog(
                f"[WARN] Candle data missing for {coin}",
                verbose_int=1,
                verbose_state=verbose,
            )
            continue
        if df.empty:
            continue
        t = len(df) - 1
        ledger_obj = Ledger.load_ledger(name, tag=ledger_cfg["tag"])
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
                process_buy_signal(
                    buy_signal=buy_res,
                    ledger=ledger_obj,
                    t=t,
                    runtime_state=state,
                    pair_code=ledger_cfg["kraken_pair"],
                    price=price,
                    ledger_name=ledger_cfg["tag"],
                    wallet_code=ledger_cfg.get("wallet_code", ""),
                    verbose=state.get("verbose", 0),
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
                    pair_code=ledger_cfg["kraken_pair"],
                    coin_amount=note.get("entry_amount", 0.0),
                    price=price,
                    ledger_name=ledger_cfg["tag"],
                    verbose=state.get("verbose", 0),
                )
                if result and not result.get("error"):
                    apply_sell_result_to_ledger(
                        ledger=ledger_obj,
                        note=note,
                        t=t,
                        result=result,
                        state=state,
                    )

        save_ledger(name, ledger_obj, tag=ledger_cfg["tag"])


def run_live(*, dry: bool = False, verbose: int = 0) -> None:
    settings = load_settings()
    runtime_states: Dict[str, Dict] = {}

    # Clear any stale buy unlock gates on startup
    for name, ledger_cfg in settings.get("ledger_settings", {}).items():
        state = build_runtime_state(
            settings,
            ledger_cfg,
            mode="live",
            prev={"verbose": verbose},
        )
        state["buy_unlock_p"] = {}
        runtime_states[name] = state

        ledger_obj = Ledger.load_ledger(name, tag=ledger_cfg["tag"])
        open_notes = ledger_obj.get_open_notes()
        total = len(open_notes)
        per_window: Dict[str, int] = {}
        last_ts = None
        for n in open_notes:
            w = n.get("window_name")
            per_window[w] = per_window.get(w, 0) + 1
            ts = n.get("created_ts")
            if ts is not None and (last_ts is None or ts > last_ts):
                last_ts = ts
        addlog(
            f"[LEDGER][OPEN] total={total} per-window={per_window}",
            verbose_int=1,
            verbose_state=verbose,
        )
        if last_ts is not None:
            addlog(
                f"[LEDGER][LAST_TS] {datetime.fromtimestamp(last_ts, tz=timezone.utc).isoformat()}",
                verbose_int=1,
                verbose_state=verbose,
            )
        else:
            addlog(
                "[LEDGER][LAST_TS] none",
                verbose_int=1,
                verbose_state=verbose,
            )

    if dry:
        addlog(
            "[LIVE] Refreshing candles (recent 720 via Kraken)...",
            verbose_int=1,
            verbose_state=verbose,
        )
        for ledger_cfg in settings.get("ledger_settings", {}).values():
            base, _ = split_tag(ledger_cfg["tag"])
            fetch_recent_coin(base)
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
        addlog(
            "[LIVE] Refreshing candles (recent 720 via Kraken)...",
            verbose_int=1,
            verbose_state=verbose,
        )
        for ledger_cfg in settings.get("ledger_settings", {}).values():
            base, _ = split_tag(ledger_cfg["tag"])
            fetch_recent_coin(base)
        addlog("[LIVE] Running top of hour", verbose_int=1, verbose_state=verbose)
        _run_iteration(settings, runtime_states, dry=dry, verbose=verbose)
