from __future__ import annotations
"""Safe account validation without placing trades."""

import os
import time
from typing import Any

import ccxt

from systems.scripts.fetch_candles import fetch_candles
from systems.scripts.kraken_utils import get_kraken_balance
from systems.scripts.account_book import AccountBook
from systems.scripts.evaluate_buy import evaluate_buy
from systems.scripts.evaluate_sell import evaluate_sell
from systems.scripts.runtime_state import build_runtime_state
from systems.utils.config import (
    load_general,
    load_coin_settings,
    load_account_settings,
    load_keys,
)
from systems.utils.time import parse_duration


def run_test(account: str, market: str, lookback: str | None = None) -> int:
    """Run a short in-memory backtest requiring at least one BUY and SELL."""

    general = load_general()
    coin_settings = load_coin_settings()
    accounts_cfg = load_account_settings()
    keys = load_keys()

    acct_cfg = accounts_cfg.get(account)
    if not acct_cfg:
        print(f"[TEST][FAIL] Account={account}\nReason: Unknown account")
        return 1
    if market not in acct_cfg.get("market settings", {}):
        print(
            f"[TEST][FAIL] Account={account} Market={market}\nReason: Unknown market"
        )
        return 1

    lookback = lookback or "72h"
    try:
        delta = parse_duration(lookback)
    except Exception:
        print(
            f"[TEST][FAIL] Account={account} Market={market}\nReason: Invalid lookback '{lookback}'"
        )
        return 1

    end_ts = int(time.time())
    start_ts = end_ts - int(delta.total_seconds())

    df = fetch_candles(market, start_ts, end_ts, source="kraken")
    if df.empty:
        print(
            f"[TEST][FAIL] Account={account} Market={market}\nReason: No recent candles in lookback"
        )
        return 1

    keypair = keys.get(account, {})
    client = ccxt.kraken(
        {
            "enableRateLimit": True,
            "apiKey": keypair.get("api_key", ""),
            "secret": keypair.get("api_secret", ""),
        }
    )

    runtime_state = build_runtime_state(
        general,
        coin_settings,
        accounts_cfg,
        account,
        market,
        mode="sim",
        client=client,
        prev={"verbose": 0},
    )
    runtime_state["mode"] = "test"
    strategy_cfg = runtime_state.get("strategy", {})

    window = int(strategy_cfg.get("window_size", 0))
    ctx: dict[str, Any] = {"account": AccountBook()}
    saw_buy = 0
    saw_sell = 0

    for t in range(window, len(df)):
        buy_res = evaluate_buy(ctx, t, df, cfg=strategy_cfg, runtime_state=runtime_state)
        if buy_res:
            saw_buy += 1
            candle = df.iloc[t]
            note = {
                "entry_price": float(candle["close"]),
                "size": float(buy_res.get("size_usd", 0.0)),
                "created_ts": int(candle.get("timestamp", end_ts)),
            }
            ctx["account"].open_note(note)

        sell_res = evaluate_sell(
            ctx,
            t,
            df,
            cfg=strategy_cfg,
            open_notes=ctx["account"].get_open_notes(),
            runtime_state=runtime_state,
        )
        if sell_res:
            saw_sell += len(sell_res)
            for note in sell_res:
                ctx["account"].close_note(note)

    features = runtime_state.get("last_features", {}).get("strategy", {})
    slope = features.get("slope", 0.0)
    vol = features.get("volatility", 0.0)

    os.environ["WS_ACCOUNT"] = account
    base, quote = market.split("/")
    try:
        balances = get_kraken_balance(quote)
    except Exception:
        balances = {}
    quote_bal = float(balances.get(quote.upper(), 0.0))
    base_bal = float(balances.get(base.upper(), 0.0))

    passed = saw_buy > 0 and saw_sell > 0
    prefix = "[TEST][PASS]" if passed else "[TEST][FAIL]"
    print(f"{prefix} Account={account} Market={market}")
    print(f"Balances: {quote.upper()}={quote_bal:.2f} {base.upper()}={base_bal:.2f}")
    print(f"Summary: buys={saw_buy} sells={saw_sell} lookback={lookback}")
    if not passed:
        reason = []
        if saw_buy == 0:
            reason.append("no BUY in lookback")
        if saw_sell == 0:
            reason.append("no SELL in lookback")
        print("Reason: " + ", ".join(reason))
    print(f"Last candle: slope={slope:.2f} vol={vol:.2f}")

    return 0 if passed else 1

