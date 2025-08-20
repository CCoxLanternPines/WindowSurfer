from __future__ import annotations
"""Safe ledger validation without placing trades."""

import os
import time
from typing import Any

import ccxt

from systems.scripts.fetch_candles import fetch_candles
from systems.scripts.kraken_utils import get_kraken_balance
from systems.scripts.ledger import Ledger
from systems.scripts.evaluate_buy import evaluate_buy
from systems.scripts.evaluate_sell import evaluate_sell
from systems.scripts.runtime_state import build_runtime_state
from systems.utils.config import (
    load_general,
    load_coin_settings,
    load_account_settings,
    load_keys,
    resolve_coin_config,
    resolve_account_market,
)


def _parse_ledger_name(name: str) -> tuple[str, str, str, str] | None:
    parts = name.split("_")
    if len(parts) < 3:
        return None
    account = parts[0]
    base = parts[1]
    quote = parts[2]
    market = f"{base}/{quote}"
    return account, base, quote, market


def run_test(ledger: str) -> int:
    """Validate ``ledger`` without mutating state or placing orders."""

    parsed = _parse_ledger_name(ledger)
    if not parsed:
        print(f"[TEST][FAIL] Ledger={ledger}\nReason: Invalid ledger name")
        return 1

    account, base, quote, market = parsed

    try:
        general = load_general()
        coin_settings = load_coin_settings()
        accounts_cfg = load_account_settings()
        keys = load_keys()
        acct_cfg = accounts_cfg.get(account)
        if not acct_cfg:
            raise ValueError(f"Unknown account '{account}'")
        if market not in acct_cfg.get("market settings", {}):
            raise ValueError(f"Unknown market '{market}'")

        strategy_cfg = {
            **resolve_coin_config(market, coin_settings),
            **resolve_account_market(account, market, accounts_cfg),
        }

        keypair = keys.get(account, {})
        client = ccxt.kraken(
            {
                "enableRateLimit": True,
                "apiKey": keypair.get("api_key", ""),
                "secret": keypair.get("api_secret", ""),
            }
        )

        now = int(time.time())
        start = now - 3 * 3600
        df = fetch_candles(market, start, now, source="kraken")
        if df.empty:
            raise RuntimeError("No recent candles fetched")

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

        window_size = int(strategy_cfg.get("window_size", 0))
        t = max(0, len(df) - window_size)

        ctx: dict[str, Any] = {"ledger": Ledger()}
        decision = "HOLD"
        buy_res = evaluate_buy(ctx, t, df, cfg=strategy_cfg, runtime_state=runtime_state)
        sell_res = evaluate_sell(
            ctx,
            t,
            df,
            cfg=strategy_cfg,
            open_notes=ctx["ledger"].get_open_notes(),
            runtime_state=runtime_state,
        )
        if buy_res:
            decision = "BUY"
        elif sell_res:
            decision = "SELL"

        features = runtime_state.get("last_features", {}).get("strategy", {})
        slope = features.get("slope", 0.0)
        vol = features.get("volatility", 0.0)

        os.environ["WS_ACCOUNT"] = account
        balances = get_kraken_balance(quote)
        quote_bal = float(balances.get(quote.upper(), 0.0))
        base_bal = float(balances.get(base.upper(), 0.0))

        print(f"[TEST][PASS] Ledger={ledger}")
        print(
            f"Balances: {quote.upper()}={quote_bal:.2f} {base.upper()}={base_bal:.2f}"
        )
        print(
            f"Decision: {decision} (slope={slope:.2f} vol={vol:.2f})"
        )
        return 0

    except Exception as exc:  # pragma: no cover - best effort logging
        print(f"[TEST][FAIL] Ledger={ledger}\nReason: {exc}")
        return 1
