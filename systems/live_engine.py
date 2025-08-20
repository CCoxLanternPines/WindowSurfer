from __future__ import annotations

"""Live engine mirroring the simulation strategy."""

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import pandas as pd
from tqdm import tqdm

import ccxt

from systems.utils.resolve_symbol import (
    to_tag,
    resolve_symbols,
    live_path_csv,
    sim_path_csv,
)
from systems.scripts.candle_cache import refresh_live_kraken_720, load_sim_for_high_low
from systems.scripts.ledger import load_ledger, save_ledger
from systems.scripts.evaluate_buy import evaluate_buy
from systems.scripts.evaluate_sell import evaluate_sell
from systems.scripts.runtime_state import build_runtime_state
from systems.scripts.trade_apply import apply_sell
from systems.scripts.execution_handler import execute_sell, process_buy_signal
from systems.utils.addlog import addlog
from systems.utils.load_config import load_config


def _run_iteration(
    cfg,
    runtime_states: Dict[str, Dict],
    hist_cache: Dict[str, tuple[float, float]],
    client,
    *,
    account_filter: str | None,
    market_filter: str | None,
    verbose: int,
) -> None:
    for acct_name, acct_cfg in cfg.get("accounts", {}).items():
        if account_filter and acct_name != account_filter:
            continue
        os.environ["WS_ACCOUNT"] = acct_name
        for market, strategy_cfg in acct_cfg.get("markets", {}).items():
            if market_filter and market != market_filter:
                continue
            symbols = resolve_symbols(client, market)
            addlog(
                f"[RESOLVE][{acct_name}][{market}] KrakenName={symbols['kraken_name']} KrakenPair={symbols['kraken_pair']} BinanceName={symbols['binance_name']}",
                verbose_int=1,
                verbose_state=verbose,
            )
            kraken_symbol = symbols["kraken_name"]
            kraken_pair = symbols["kraken_pair"]
            tag = to_tag(kraken_symbol)
            file_tag = kraken_symbol.replace("/", "_")
            ledger_name = f"{acct_name}_{file_tag}"
            base = kraken_symbol.split("/")[0]
            refresh_live_kraken_720(kraken_symbol)
            live_file = live_path_csv(tag)
            if not Path(live_file).exists():
                print(
                    f"[ERROR] Missing data file: {live_file}. Run: python bot.py --mode fetch --account {acct_name} --market {market}"
                )
                raise SystemExit(1)
            df = pd.read_csv(live_file)
            ts_col = next(
                (
                    c
                    for c in df.columns
                    if str(c).lower() in ("timestamp", "time", "date")
                ),
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
                "%Y-%m-%dT%H:%M:%SZ",
            )
            print(f"[DATA][LIVE] file={live_file} rows={len(df)} last={last_iso}")
            if tag not in hist_cache:
                sim_file = sim_path_csv(tag)
                if not Path(sim_file).exists():
                    print(
                        f"[ERROR] Missing data file: {sim_file}. Run: python bot.py --mode fetch --account {acct_name} --market {market}"
                    )
                    raise SystemExit(1)
                hist_low, hist_high = load_sim_for_high_low(tag)
                hist_cache[tag] = (hist_low, hist_high)
                print(
                    f"[STATS][LIVE] hist_low={hist_low:.2f} hist_high={hist_high:.2f} from={sim_file}"
                )
            hist_low, hist_high = hist_cache[tag]
            t = len(df) - 1
            ledger_obj = load_ledger(ledger_name, tag=file_tag)
            prev = runtime_states.get(ledger_name, {"verbose": verbose})
            state = build_runtime_state(
                cfg,
                market,
                strategy_cfg,
                mode="live",
                prev=prev,
                client=client,
            )
            state["mode"] = "live"
            state["symbol"] = tag
            state["hist_low"] = hist_low
            state["hist_high"] = hist_high
            runtime_states[ledger_name] = state

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
                process_buy_signal(
                    buy_signal=buy_res,
                    ledger=ledger_obj,
                    t=t,
                    runtime_state=state,
                    pair_code=kraken_pair,
                    price=price,
                    ledger_name=tag,
                    wallet_code=base,
                    verbose=state.get("verbose", 0),
                )
            sell_orders = evaluate_sell(
                {"ledger": ledger_obj},
                t,
                df,
                cfg=strategy_cfg,
                open_notes=ledger_obj.get_open_notes(),
                runtime_state=state,
            )
            for order in sell_orders:
                note = order["note"]
                amt = order["sell_amount"]
                mode = order.get("sell_mode", "normal")
                entry_price = note.get("entry_price", 0.0)
                result = execute_sell(
                    None,
                    pair_code=kraken_pair,
                    coin_amount=amt,
                    price=price,
                    ledger_name=tag,
                    wallet_code=base,
                    verbose=state.get("verbose", 0),
                )
                if result and not result.get("error"):
                    if amt >= note.get("entry_amount", 0.0) - 1e-9:
                        note["sell_mode"] = mode
                        apply_sell(
                            ledger=ledger_obj,
                            note=note,
                            t=t,
                            result=result,
                            state=state,
                        )
                    else:
                        partial = note.copy()
                        partial["entry_amount"] = amt
                        partial["entry_usdt"] = amt * entry_price
                        partial["sell_mode"] = mode
                        ledger_obj.open_note(partial)
                        apply_sell(
                            ledger=ledger_obj,
                            note=partial,
                            t=t,
                            result=result,
                            state=state,
                        )
                        note["entry_amount"] -= amt
                        note["entry_usdt"] -= amt * entry_price

            save_ledger(ledger_name, ledger_obj, tag=file_tag)


def run_live(
    *, account: str, market: str | None = None, dry: bool = False, verbose: int = 0
) -> None:
    cfg = load_config()
    runtime_states: Dict[str, Dict] = {}
    hist_cache: Dict[str, tuple[float, float]] = {}
    client = ccxt.kraken()

    for acct_name, acct_cfg in cfg.get("accounts", {}).items():
        if account and acct_name != account:
            continue
        os.environ["WS_ACCOUNT"] = acct_name
        for mkt, strat in acct_cfg.get("markets", {}).items():
            if market and mkt != market:
                continue
            symbols = resolve_symbols(client, mkt)
            addlog(
                f"[RESOLVE][{acct_name}][{mkt}] KrakenName={symbols['kraken_name']} KrakenPair={symbols['kraken_pair']} BinanceName={symbols['binance_name']}",
                verbose_int=1,
                verbose_state=verbose,
            )
            tag = to_tag(symbols["kraken_name"])
            file_tag = symbols["kraken_name"].replace("/", "_")
            ledger_name = f"{acct_name}_{file_tag}"
            state = build_runtime_state(
                cfg,
                mkt,
                strat,
                mode="live",
                prev={"verbose": verbose},
                client=client,
            )
            state["buy_unlock_p"] = {}
            state["symbol"] = tag
            runtime_states[ledger_name] = state

            ledger_obj = load_ledger(ledger_name, tag=file_tag)
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
            cfg,
            runtime_states,
            hist_cache,
            client,
            account_filter=account,
            market_filter=market,
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
            cfg,
            runtime_states,
            hist_cache,
            client,
            account_filter=account,
            market_filter=market,
            verbose=verbose,
        )
