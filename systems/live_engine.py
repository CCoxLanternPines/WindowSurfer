from __future__ import annotations

"""Live engine mirroring the simulation strategy."""

import time
from datetime import datetime, timezone
from typing import Dict

import pandas as pd
from tqdm import tqdm

from systems.scripts.candle_cache import (
    tag_from_symbol,
    live_path_csv,
    load_sim_for_high_low,
    hard_refresh_live_720,
    last_closed_hour_ts,
    sim_path_csv,
)
from systems.scripts.ledger import load_ledger, save_ledger
from systems.scripts.evaluate_buy import evaluate_buy
from systems.scripts.evaluate_sell import evaluate_sell
from systems.scripts.runtime_state import build_runtime_state
from systems.scripts.trade_apply import apply_sell
from systems.scripts.execution_handler import execute_sell, process_buy_signal
from systems.scripts.strategy_jackpot import (
    init_jackpot,
    on_buy_drip,
    maybe_periodic_jackpot_buy,
    maybe_cashout_jackpot,
)
from systems.utils.addlog import addlog
from systems.utils.config import load_settings


def _run_iteration(settings, runtime_states, *, dry: bool, verbose: int) -> None:
    tag_cache: Dict[str, pd.DataFrame] = {}
    hist_cache: Dict[str, tuple[float, float]] = {}
    for name, ledger_cfg in settings.get("ledger_settings", {}).items():
        symbol = ledger_cfg["kraken_name"]
        tag = tag_from_symbol(symbol)
        window_settings = ledger_cfg.get("window_settings", {})
        if tag not in tag_cache:
            end_ts = last_closed_hour_ts(int(time.time()))
            iso = datetime.fromtimestamp(end_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            print(f"[LIVE][REFRESH] symbol={symbol} tag={tag} window=720h end={iso}")
            hard_refresh_live_720(symbol)
            df_live = pd.read_csv(live_path_csv(tag))
            hist_low, hist_high = load_sim_for_high_low(tag)
            print(
                f"[LIVE][STATS] hist_low={hist_low:.2f} hist_high={hist_high:.2f} source={sim_path_csv(tag)}"
            )
            tag_cache[tag] = df_live
            hist_cache[tag] = (hist_low, hist_high)
        df = tag_cache[tag]
        hist_low, hist_high = hist_cache[tag]
        if df.empty:
            continue
        t = len(df) - 1
        ledger_obj = load_ledger(name, tag=ledger_cfg["tag"])
        prev = runtime_states.get(name, {"verbose": verbose})
        state = build_runtime_state(
            settings,
            ledger_cfg,
            mode="live",
            prev=prev,
        )
        state["mode"] = "live"
        state["hist_low"] = hist_low
        state["hist_high"] = hist_high
        runtime_states[name] = state
        init_jackpot(state, ledger_cfg, df)
        j = state.get("jackpot", {})
        if j.get("enabled"):
            j["notes_open"] = [n for n in ledger_obj.get_open_notes() if n.get("kind") == "jackpot"]

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
                buy_res["size_usd"] = on_buy_drip(state, buy_res["size_usd"])
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
                    apply_sell(
                        ledger=ledger_obj,
                        note=note,
                        t=t,
                        result=result,
                        state=state,
                )

        ctx_j = {
            "ledger": ledger_obj,
            "pair_code": ledger_cfg["kraken_pair"],
            "wallet_code": ledger_cfg.get("wallet_code", ""),
            "verbosity": state.get("verbose", 0),
        }
        maybe_periodic_jackpot_buy(
            ctx_j,
            state,
            t,
            df,
            price,
            state.get("limits", {}),
            ledger_cfg["tag"],
        )
        maybe_cashout_jackpot(
            ctx_j,
            state,
            t,
            df,
            price,
            state.get("limits", {}),
            ledger_cfg["tag"],
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

        ledger_obj = load_ledger(name, tag=ledger_cfg["tag"])
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
