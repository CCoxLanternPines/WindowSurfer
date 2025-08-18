from __future__ import annotations

"""Live engine mirroring the simulation strategy."""

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import pandas as pd
from tqdm import tqdm

from systems.utils.resolve_symbol import (
    to_tag,
    resolve_ccxt_symbols,
    live_path_csv,
    sim_path_csv,
)
from systems.scripts.candle_cache import (
    refresh_live_kraken_720,
    load_sim_for_high_low,
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


def _run_iteration(
    settings,
    runtime_states,
    hist_cache,
    *,
    ledger_filter: str | None,
    verbose: int,
) -> None:
    for name, ledger_cfg in settings.get("ledger_settings", {}).items():
        if ledger_filter and name != ledger_filter:
            continue
        kraken_symbol, _ = resolve_ccxt_symbols(settings, name)
        tag = to_tag(kraken_symbol)
        strategy_cfg = settings.get("general_settings", {}).get("strategy_settings", {})
        refresh_live_kraken_720(kraken_symbol)
        live_file = live_path_csv(tag)
        if not Path(live_file).exists():
            print(
                f"[ERROR] Missing data file: {live_file}. Run: python bot.py --mode fetch --ledger {name}"
            )
            raise SystemExit(1)
        df = pd.read_csv(live_file)
        ts_col = next(
            (c for c in df.columns if str(c).lower() in ("timestamp", "time", "date")),
            None,
        )
        if ts_col is None:
            print(f"[ERROR] Missing timestamp column in {live_file}")
            raise SystemExit(1)
        df[ts_col] = pd.to_numeric(df[ts_col], errors="coerce")
        df = df.dropna(subset=[ts_col])
        if df.empty:
            continue
        last_ts = int(df[ts_col].iloc[-1])
        last_iso = datetime.fromtimestamp(last_ts, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        print(f"[DATA][LIVE] file={live_file} rows={len(df)} last={last_iso}")
        if tag not in hist_cache:
            sim_file = sim_path_csv(tag)
            if not Path(sim_file).exists():
                print(
                    f"[ERROR] Missing data file: {sim_file}. Run: python bot.py --mode fetch --ledger {name}"
                )
                raise SystemExit(1)
            hist_low, hist_high = load_sim_for_high_low(tag)
            hist_cache[tag] = (hist_low, hist_high)
            print(
                f"[STATS][LIVE] hist_low={hist_low:.2f} hist_high={hist_high:.2f} from={sim_file}"
            )
        hist_low, hist_high = hist_cache[tag]
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
        state["symbol"] = ledger_cfg.get("tag", "")
        state["hist_low"] = hist_low
        state["hist_high"] = hist_high
        runtime_states[name] = state
        init_jackpot(state, ledger_cfg, df)
        j = state.get("jackpot", {})
        if j.get("enabled"):
            j["notes_open"] = [n for n in ledger_obj.get_open_notes() if n.get("kind") == "jackpot"]

        price = float(df.iloc[t]["close"])
        ctx = {"ledger": ledger_obj}
        buy_res = evaluate_buy(
            ctx,
            t,
            df,
            cfg=strategy_cfg,
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
        # evaluate_sell relies on pressures updated by evaluate_buy
        sell_notes = evaluate_sell(
            ctx,
            t,
            df,
            cfg=strategy_cfg,
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


def run_live(*, ledger: str | None = None, dry: bool = False, verbose: int = 0) -> None:
    settings = load_settings()
    runtime_states: Dict[str, Dict] = {}
    hist_cache: Dict[str, tuple[float, float]] = {}

    # Clear any stale buy unlock gates on startup
    for name, ledger_cfg in settings.get("ledger_settings", {}).items():
        if ledger and name != ledger:
            continue
        state = build_runtime_state(
            settings,
            ledger_cfg,
            mode="live",
            prev={"verbose": verbose},
        )
        state["buy_unlock_p"] = {}
        state["symbol"] = ledger_cfg.get("tag", "")
        runtime_states[name] = state

        ledger_obj = load_ledger(name, tag=ledger_cfg["tag"])
        open_notes = ledger_obj.get_open_notes()
        total = len(open_notes)
        last_ts = None
        for n in open_notes:
            ts = n.get("created_ts")
            if ts is not None and (last_ts is None or ts > last_ts):
                last_ts = ts
        addlog(
            f"[LEDGER][OPEN] total={total}",
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
        _run_iteration(
            settings,
            runtime_states,
            hist_cache,
            ledger_filter=ledger,
            verbose=verbose,
        )
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
        _run_iteration(
            settings,
            runtime_states,
            hist_cache,
            ledger_filter=ledger,
            verbose=verbose,
        )
