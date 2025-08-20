from __future__ import annotations

"""Live engine mirroring the simulation strategy."""

import os
from datetime import datetime, timezone
from typing import Dict

import ccxt
from tqdm import tqdm

from systems.utils.resolve_symbol import (
    to_tag,
    resolve_symbols,
    candle_filename,
)
from systems.scripts.fetch_candles import fetch_kraken_last_n_hours_1h
from systems.scripts.ledger import load_ledger, save_ledger
from systems.scripts.evaluate_buy import evaluate_buy
from systems.scripts.evaluate_sell import evaluate_sell
from systems.scripts.runtime_state import build_runtime_state
from systems.scripts.trade_apply import apply_sell
from systems.scripts.execution_handler import execute_sell, process_buy_signal
from systems.scripts.candle_loader import load_candles_df
from systems.utils.addlog import addlog
from systems.utils.load_config import load_config


def _run_iteration(
    cfg,
    runtime_states: Dict[str, Dict],
    hist_cache: Dict[str, tuple[float, float]],
    *,
    account_filter: str | None,
    market_filter: str | None,
    verbose: int,
) -> None:
    for acct_name, acct_cfg in cfg.get("accounts", {}).items():
        if account_filter and acct_name != account_filter:
            continue
        os.environ["WS_ACCOUNT"] = acct_name
        client = ccxt.kraken(
            {
                "enableRateLimit": True,
                "apiKey": acct_cfg.get("api_key", ""),
                "secret": acct_cfg.get("api_secret", ""),
            }
        )
        for market, strategy_cfg in acct_cfg.get("markets", {}).items():
            if market_filter and market != market_filter:
                continue
            addlog(
                f"[RUN][{acct_name}][{market}]",
                verbose_int=1,
                verbose_state=verbose,
            )
            symbols = resolve_symbols(client, market)
            kraken_name = symbols["kraken_name"]
            kraken_pair = symbols["kraken_pair"]
            binance_name = symbols["binance_name"]
            tag = to_tag(kraken_name)
            file_tag = market.replace("/", "_")
            ledger_name = f"{acct_name}_{file_tag}"
            base = kraken_name.split("/")[0]
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
                "%Y-%m-%dT%H:%M:%SZ",
            )
            print(f"[DATA][LIVE] file={live_file} rows={len(df)} last={last_iso}")
            if ledger_name not in hist_cache:
                df_sim, _ = load_candles_df(acct_name, market, verbose=verbose)
                hist_low = float(df_sim["low"].min())
                hist_high = float(df_sim["high"].max())
                hist_cache[ledger_name] = (hist_low, hist_high)
                sim_file = candle_filename(acct_name, market)
                print(
                    f"[STATS][LIVE] hist_low={hist_low:.2f} hist_high={hist_high:.2f} from={sim_file}"
                )
            hist_low, hist_high = hist_cache[ledger_name]
            t = len(df) - 1
            ledger_obj = load_ledger(ledger_name, tag=file_tag)
            prev = runtime_states.get(ledger_name, {"verbose": verbose})
            state = build_runtime_state(
                cfg,
                market,
                strategy_cfg,
                mode="sim",
                client=client,
                prev=prev,
            )
            state["mode"] = "live"
            state["symbol"] = tag
            state["hist_low"] = hist_low
            state["hist_high"] = hist_high
            state["capital"] = ledger_obj.get_metadata().get("capital", state.get("capital", 0.0))
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
            sell_notes = evaluate_sell(
                {"ledger": ledger_obj},
                t,
                df,
                cfg=strategy_cfg,
                open_notes=ledger_obj.get_open_notes(),
                runtime_state=state,
            )
            for note in sell_notes:
                ts = int(df.iloc[t]["timestamp"])
                result = execute_sell(
                    None,
                    pair_code=kraken_pair,
                    coin_amount=note.get("entry_amount", 0.0),
                    price=price,
                    ledger_name=tag,
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

            ledger_obj.set_metadata({"capital": state.get("capital", 0.0)})
            save_ledger(ledger_name, ledger_obj, tag=file_tag)


def run_live(
    *,
    account: str | None = None,
    market: str | None = None,
    all_accounts: bool = False,
    dry: bool = False,
    verbose: int = 0,
) -> None:
    cfg = load_config()
    addlog(
        "[PARITY] Running in live mode — strategy knobs identical, only execution differs",
        verbose_int=1,
        verbose_state=verbose,
    )
    runtime_states: Dict[str, Dict] = {}
    hist_cache: Dict[str, tuple[float, float]] = {}

    targets = (
        cfg.get("accounts", {}).keys()
        if (all_accounts or not account)
        else [account]
    )

    for acct_name in targets:
        acct_cfg = cfg.get("accounts", {}).get(acct_name)
        if not acct_cfg:
            continue
        os.environ["WS_ACCOUNT"] = acct_name
        client = ccxt.kraken(
            {
                "enableRateLimit": True,
                "apiKey": acct_cfg.get("api_key", ""),
                "secret": acct_cfg.get("api_secret", ""),
            }
        )
        for mkt, strat in acct_cfg.get("markets", {}).items():
            if market and mkt != market:
                continue
            addlog(
                f"[RUN][{acct_name}][{mkt}]",
                verbose_int=1,
                verbose_state=verbose,
            )
            symbols = resolve_symbols(client, mkt)
            kraken_name = symbols["kraken_name"]
            kraken_pair = symbols["kraken_pair"]
            binance_name = symbols["binance_name"]
            addlog(
                f"[RESOLVE][{acct_name}][{mkt}] KrakenName={kraken_name} KrakenPair={kraken_pair} BinanceName={binance_name}",
                verbose_int=1,
                verbose_state=verbose,
            )
            tag = to_tag(kraken_name)
            file_tag = mkt.replace("/", "_")
            ledger_name = f"{acct_name}_{file_tag}"
            state = build_runtime_state(
                cfg,
                mkt,
                strat,
                mode="sim",
                client=client,
                prev={"verbose": verbose},
            )
            state["mode"] = "live"
            state["buy_unlock_p"] = {}
            state["symbol"] = tag
            ledger_obj = load_ledger(ledger_name, tag=file_tag)
            state["capital"] = ledger_obj.get_metadata().get("capital", state.get("capital", 0.0))
            runtime_states[ledger_name] = state

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

    account_filter = None if (all_accounts or not account) else account
    if dry:
        _run_iteration(
            cfg,
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
        addlog("[LIVE] Running top of hour", verbose_int=1, verbose_state=verbose)
        _run_iteration(
            cfg,
            runtime_states,
            hist_cache,
            account_filter=account_filter,
            market_filter=market,
            verbose=verbose,
        )
