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
from systems.scripts.smoke_test import run_smoke_test
from systems.utils.addlog import addlog
from systems.utils.config import load_settings
from systems.utils.resolve_symbol import (
    load_pair_cache,
    resolve_wallet_codes,
    resolve_ccxt_symbols,
)


def _run_iteration(
    settings,
    runtime_states,
    *,
    dry: bool,
    verbose: int,
    smoke: bool = False,
    smoke_save: bool = False,
) -> None:
    cache = load_pair_cache()
    for name, ledger_cfg in settings.get("ledger_settings", {}).items():
        coin = ledger_cfg["coin"]
        fiat = ledger_cfg["fiat"]
        window_settings = ledger_cfg.get("window_settings", {})
        try:
            df = fetch_candles(coin, fiat)
        except FileNotFoundError:
            addlog(
                f"[WARN] Candle data missing for {coin}/{fiat}",
                verbose_int=1,
                verbose_state=verbose,
            )
            continue
        if df.empty:
            continue
        t = len(df) - 1
        ledger_obj = Ledger.load_ledger(tag=name)
        prev = runtime_states.get(name, {"verbose": verbose})
        state = build_runtime_state(
            settings,
            ledger_cfg,
            mode="live",
            ledger_name=name,
            prev=prev,
        )
        runtime_states[name] = state
        codes = resolve_wallet_codes(coin, fiat, cache, state.get("verbose", 0))
        wallet_code = ledger_cfg.get("wallet_code") or codes["base_wallet_code"]
        fiat_code = codes["quote_wallet_code"]
        syms = resolve_ccxt_symbols(coin, fiat, cache, state.get("verbose", 0))
        pair_code = syms["kraken_pair"]

        price = float(df.iloc[t]["close"])
        candle = df.iloc[t].to_dict()
        did_buy = False
        did_sell = False
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
                    pair_code=pair_code,
                    fiat_symbol=fiat_code,
                    price=price,
                    amount_usd=buy_res["size_usd"],
                    ledger_name=name,
                    wallet_code=wallet_code,
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
                    did_buy = True

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
                    pair_code=pair_code,
                    fiat_symbol=fiat_code,
                    coin_amount=note.get("entry_amount", 0.0),
                    price=price,
                    ledger_name=name,
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
                    did_sell = True

        if dry and smoke and not did_buy and not did_sell:
            run_smoke_test(
                ledger_name=name,
                ledger_cfg=ledger_cfg,
                settings=settings,
                candle=candle,
                state=state,
                save=smoke_save,
                verbose=verbose,
            )

        save_ledger(name, ledger_obj)


def run_live(
    *,
    dry: bool = False,
    verbose: int = 0,
    smoke: bool = False,
    smoke_save: bool = False,
    replay_hours: int | None = None,
) -> None:
    settings = load_settings()
    runtime_states: Dict[str, Dict] = {}

    if dry:
        _run_iteration(
            settings,
            runtime_states,
            dry=dry,
            verbose=verbose,
            smoke=smoke,
            smoke_save=smoke_save,
        )
        if replay_hours:
            from systems.sim_engine import run_simulation

            for name, ledger_cfg in settings.get("ledger_settings", {}).items():
                coin = ledger_cfg["coin"]
                fiat = ledger_cfg["fiat"]
                try:
                    df = fetch_candles(coin, fiat)
                except FileNotFoundError:
                    continue
                if df.empty:
                    continue
                end_idx = len(df) - 1
                start_idx = max(0, end_idx - replay_hours + 1)
                summary = run_simulation(
                    ledger=name,
                    verbose=0,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    save_ledger_file=False,
                    progress=False,
                )
                addlog(
                    f"[REPLAY][N={replay_hours}h] buys={summary['buys']} sells={summary['sells']} pnl=${summary['realized_gain']:.2f} open=${summary['open_value']:.2f}",
                    verbose_int=1,
                    verbose_state=verbose,
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
            dry=dry,
            verbose=verbose,
            smoke=smoke,
            smoke_save=smoke_save,
        )
