from __future__ import annotations

"""Live engine mirroring the simulation strategy."""

import os
import time
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

import ccxt
from tqdm import tqdm

from systems.utils.resolve_symbol import to_tag, resolve_symbols, candle_filename
from systems.scripts.fetch_candles import fetch_kraken_last_n_hours_1h
from systems.scripts.ledger import load_ledger as load_state_ledger, save_ledger as save_state_ledger
from systems.scripts.evaluate_buy import evaluate_buy
from systems.scripts.evaluate_sell import evaluate_sell
from systems.scripts.runtime_state import build_runtime_state
from systems.scripts.trade_apply import apply_sell
from systems.scripts.execution_handler import execute_sell, process_buy_signal
from systems.scripts.candle_loader import load_candles_df
from systems.utils.trade_logger import init_logger as init_trade_logger, record_event
from systems.utils.config import (
    load_general,
    load_coin_settings,
    load_account_settings,
    load_keys,
    resolve_coin_config,
    resolve_account_market,
)
from systems.utils.ledger import (
    init_ledger as ledger_init,
    append_entry as ledger_append,
    save_ledger as ledger_save,
    load_ledger as ledger_load,
)


def _run_iteration(
    general,
    coin_settings,
    accounts_cfg,
    keys,
    runtime_states: Dict[str, Dict],
    hist_cache: Dict[str, tuple[float, float]],
    *,
    account_filter: str | None,
    market_filter: str | None,
    verbose: int,
) -> None:
    for acct_name, acct_cfg in accounts_cfg.items():
        if account_filter and acct_name != account_filter:
            continue

        if not acct_cfg.get("is_live", False):
            print(f"[SKIP] {acct_name} (is_live = false)")
            continue

        keypair = keys.get(acct_name, {})
        client = ccxt.kraken(
            {
                "enableRateLimit": True,
                "apiKey": keypair.get("api_key", ""),
                "secret": keypair.get("api_secret", ""),
            }
        )

        for market in acct_cfg.get("market settings", {}).keys():
            if market_filter and market != market_filter:
                continue

            strategy_cfg = {
                **resolve_coin_config(market, coin_settings),
                **resolve_account_market(acct_name, market, accounts_cfg),
            }

            symbols = resolve_symbols(client, market)
            kraken_name = symbols["kraken_name"]
            kraken_pair = symbols["kraken_pair"]
            binance_name = symbols["binance_name"]
            tag = to_tag(kraken_name)
            file_tag = market.replace("/", "_")
            ledger_name = f"{acct_name}_{file_tag}"

            # init log file for this ledger
            init_trade_logger(ledger_name)

            # load state ledger for open/closed notes separately
            state_ledger_name = f"{ledger_name}_state"
            ledger_obj = load_state_ledger(state_ledger_name, tag=f"{file_tag}_state")

            # load or init trading ledger for reporting
            ledger_path = Path("data") / "ledgers" / f"{ledger_name}.json"
            trade_ledger = ledger_load(str(ledger_path)) or ledger_init(acct_name, market, "live")

            # refresh last 720h from kraken
            live_file = candle_filename(acct_name, market, live=True)
            df_live = fetch_kraken_last_n_hours_1h(kraken_name, n=720)
            tmp_live = live_file + ".tmp"
            os.makedirs(os.path.dirname(live_file), exist_ok=True)
            df_live.to_csv(tmp_live, index=False)
            os.replace(tmp_live, live_file)

            df, _ = load_candles_df(acct_name, market, live=True, verbose=verbose)
            if df.empty:
                continue

            last_ts = int(df["timestamp"].iloc[-1])
            last_iso = datetime.fromtimestamp(last_ts, tz=timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            print(f"[DATA][LIVE] file={live_file} rows={len(df)} last={last_iso}")

            if ledger_name not in hist_cache:
                df_sim, _ = load_candles_df(acct_name, market, verbose=verbose)
                hist_low = float(df_sim["low"].min())
                hist_high = float(df_sim["high"].max())
                hist_cache[ledger_name] = (hist_low, hist_high)
                print(
                    f"[STATS][LIVE] hist_low={hist_low:.2f} hist_high={hist_high:.2f}"
                )
            hist_low, hist_high = hist_cache[ledger_name]

            t = len(df) - 1
            prev = runtime_states.get(ledger_name, {"verbose": verbose})
            state = build_runtime_state(
                general,
                coin_settings,
                accounts_cfg,
                acct_name,
                market,
                mode="live",
                client=client,
                prev=prev,
            )
            state["mode"] = "live"
            state["symbol"] = tag
            state["hist_low"] = hist_low
            state["hist_high"] = hist_high
            state["capital"] = ledger_obj.get_metadata().get(
                "capital", state.get("capital", 0.0)
            )
            runtime_states[ledger_name] = state
            strategy_cfg = state.get("strategy", {})

            price = float(df.iloc[t]["close"])
            ctx = {"ledger": ledger_obj}
            decision = "HOLD"
            trades_log: list[dict[str, Any]] = []
            size_usd = 0.0
            entry_price_val = 0.0
            roi_val = 0.0

            # BUY
            buy_res = evaluate_buy(
                ctx,
                t,
                df,
                cfg=strategy_cfg,
                runtime_state=state,
            )
            if buy_res:
                buy_result = process_buy_signal(
                    buy_signal=buy_res,
                    ledger=ledger_obj,
                    t=t,
                    runtime_state=state,
                    pair_code=kraken_pair,
                    price=price,
                    ledger_name=tag,
                    wallet_code=kraken_name.split("/")[0],
                    verbose=state.get("verbose", 0),
                )
                decision = "BUY"
                size_usd = buy_res.get("size_usd", 0.0)
                entry_price_val = (buy_result or {}).get("avg_price", price)
                trades_log.append(
                    {
                        "action": "BUY",
                        "amount": buy_res.get("size_usd", 0.0),
                        "price": (buy_result or {}).get("avg_price", 0.0),
                        "note_id": f"{buy_res.get('window_name','strategy')}-{t}",
                    }
                )

            # SELL
            sell_notes = evaluate_sell(
                {"ledger": ledger_obj},
                t,
                df,
                cfg=strategy_cfg,
                open_notes=ledger_obj.get_open_notes(),
                runtime_state=state,
            )
            if sell_notes:
                decision = "FLAT" if any(
                    n.get("sell_mode") == "flat" for n in sell_notes
                ) else "SELL"

            for note in sell_notes:
                result = execute_sell(
                    None,
                    pair_code=kraken_pair,
                    coin_amount=note.get("entry_amount", 0.0),
                    price=price,
                    ledger_name=tag,
                    verbose=state.get("verbose", 0),
                )
                if result and not result.get("error"):
                    closed = apply_sell(
                        ledger=ledger_obj,
                        note=note,
                        t=t,
                        result=result,
                        state=state,
                    )
                    trades_log.append(
                        {
                            "action": "SELL",
                            "amount": result.get("filled_amount", 0.0)
                            * result.get("avg_price", 0.0),
                            "price": result.get("avg_price", 0.0),
                            "note_id": closed.get("id"),
                        }
                    )
            if sell_notes:
                total_cost = sum(
                    n.get("entry_amount", 0.0) * n.get("entry_price", 0.0)
                    for n in sell_notes
                )
                total_amount = sum(n.get("entry_amount", 0.0) for n in sell_notes)
                entry_price_val = (total_cost / total_amount) if total_amount else 0.0
                size_usd = total_cost
                roi_val = ((price - entry_price_val) / entry_price_val) if entry_price_val else 0.0

            ledger_obj.set_metadata({"capital": state.get("capital", 0.0)})
            save_state_ledger(state_ledger_name, ledger_obj, tag=f"{file_tag}_state")

            # record structured event
            features = state.get("last_features", {}).get("strategy", {})
            pressures = state.get("pressures", {})
            event = {
                "timestamp": last_iso,
                "ledger": ledger_name,
                "pair": tag,
                "window": f"{strategy_cfg.get('window_size', 0)}h",
                "decision": decision,
                "features": {
                    "close": float(df.iloc[t]["close"]),
                    "slope": features.get("slope"),
                    "volatility": features.get("volatility"),
                    "buy_pressure": pressures.get("buy", {}).get("strategy", 0.0),
                    "sell_pressure": pressures.get("sell", {}).get("strategy", 0.0),
                    "buy_trigger": strategy_cfg.get("buy_trigger", 0.0),
                    "sell_trigger": strategy_cfg.get("sell_trigger", 0.0),
                },
                "trades": trades_log,
            }
            ledger_entry = {
                "candle_idx": t,
                "timestamp": last_ts,
                "side": "BUY" if decision == "BUY" else ("SELL" if sell_notes else "PASS"),
                "price": price,
                "size_usd": size_usd,
                "entry_price": entry_price_val,
                "roi": roi_val,
                "pressure_buy": pressures.get("buy", {}).get("strategy", 0.0),
                "pressure_sell": pressures.get("sell", {}).get("strategy", 0.0),
                "features": features,
            }
            ledger_append(trade_ledger, ledger_entry)
            ledger_save(trade_ledger, str(ledger_path))
            record_event(event)


def run_live(
    *,
    account: str | None = None,
    market: str | None = None,
    all_accounts: bool = False,
    dry: bool = False,
    verbose: int = 0,
) -> None:
    general = load_general()
    coin_settings = load_coin_settings()
    accounts_cfg = load_account_settings()
    keys = load_keys()
    runtime_states: Dict[str, Dict] = {}
    hist_cache: Dict[str, tuple[float, float]] = {}

    targets = (
        accounts_cfg.keys() if (all_accounts or not account) else [account]
    )

    for acct_name in targets:
        acct_cfg = accounts_cfg.get(acct_name)
        if not acct_cfg:
            continue
        if not acct_cfg.get("is_live", False):
            print(f"[SKIP] {acct_name} (is_live = false)")
            continue
        for mkt in acct_cfg.get("market settings", {}).keys():
            if market and mkt != market:
                continue
            ledger_name = f"{acct_name}_{mkt.replace('/','_')}"
            init_trade_logger(ledger_name)

    account_filter = None if (all_accounts or not account) else account
    if dry:
        _run_iteration(
            general,
            coin_settings,
            accounts_cfg,
            keys,
            runtime_states,
            hist_cache,
            account_filter=account_filter,
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
        print("[LIVE] Running top of hour")
        _run_iteration(
            general,
            coin_settings,
            accounts_cfg,
            keys,
            runtime_states,
            hist_cache,
            account_filter=account_filter,
            market_filter=market,
            verbose=verbose,
        )
